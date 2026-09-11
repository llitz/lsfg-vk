/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "instance.hpp"
#include "lsfg-vk-common/helpers/errors.hpp"
#include "lsfg-vk-common/helpers/pointers.hpp"
#include "lsfg-vk-common/vulkan/vulkan.hpp"
#include "swapchain.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <vulkan/vk_layer.h>
#include <vulkan/vulkan_core.h>

using namespace lsfgvk::layer;

    // redirect the layer's std::cerr to a file when LSFGVK_LOG_FILE is set,
    // so diagnostics can be captured regardless of how the game is launched
    // (e.g. Lutris, where the game's stderr is not visible). Wine's WINEDEBUG
    // writes fd 2 directly and is unaffected.
    void init_file_logging() {
        const char* path = std::getenv("LSFGVK_LOG_FILE");
        if (!path)
            return;
        static std::ofstream ofs(path, std::ios::app);
        if (ofs)
            std::cerr.rdbuf(ofs.rdbuf());
    }

namespace {
    // global layer info initialized at layer negotiation
    struct LayerInfo {
        std::unordered_map<std::string, PFN_vkVoidFunction> map; //!< function pointer override map
        PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
        std::unordered_map<VkInstance, PFN_vkGetInstanceProcAddr> instanceGipa; //!< per-instance next GIPA
        PFN_GetPhysicalDeviceProcAddr nextGetPhysicalDeviceProcAddr;
        std::mutex mutex; //!< guards instanceGipa + nextGetPhysicalDeviceProcAddr

        Root root;
    }* layer_info; // NOLINT (global variable)

    // instance-wide info initialized at instance creation(s)
    struct InstanceInfo {
        std::vector<VkInstance> handles; // there may be several instances
        vk::VulkanInstanceFuncs funcs;
        std::unordered_map<VkDevice, PFN_vkGetDeviceProcAddr> deviceGdpa; //!< per-device next GDPA
        std::mutex mutex; //!< guards deviceGdpa, devices, handles

        std::unordered_map<VkDevice, vk::Vulkan> devices;
        std::unordered_map<VkSwapchainKHR, ls::R<vk::Vulkan>> swapchains;
        std::unordered_map<VkSwapchainKHR, SwapchainInfo> swapchainInfos;
        std::unordered_set<VkSwapchainKHR> retiredSwapchains;
        std::unordered_map<VkSwapchainKHR, size_t> pendingMultiplier;
        std::unordered_map<VkSwapchainKHR, size_t> loggedDeferredMultiplier;
    }* instance_info; // NOLINT (global variable)
    std::mutex instance_info_mutex; //!< guards instance_info creation/destruction

    // create instance
    VkResult myvkCreateInstance(
            const VkInstanceCreateInfo* info,
            const VkAllocationCallbacks* alloc,
            VkInstance* instance) {
        // apply layer chaining
        auto* layerInfo = reinterpret_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(info->pNext));
        while (layerInfo && (layerInfo->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO
                || layerInfo->function != VK_LAYER_LINK_INFO)) {
            layerInfo = reinterpret_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(layerInfo->pNext));
        }
        if (!layerInfo) {
            std::cerr << "lsfg-vk: no layer info found in pNext chain, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto* linkInfo = layerInfo->u.pLayerInfo;
        if (!linkInfo) {
            std::cerr << "lsfg-vk: link info is null, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        layer_info->GetInstanceProcAddr = linkInfo->pfnNextGetInstanceProcAddr;
        if (!layer_info->GetInstanceProcAddr) {
            std::cerr << "lsfg-vk: next layer's vkGetInstanceProcAddr is null, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        layerInfo->u.pLayerInfo = linkInfo->pNext; // advance for next layer

        // create instance
        const bool wsi = requests_wsi_surface(info->ppEnabledExtensionNames,
            info->enabledExtensionCount);

        auto* vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
            layer_info->GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
        if (!vkCreateInstance) {
            std::cerr << "lsfg-vk: failed to get next layer's vkCreateInstance, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        try {
            if (!wsi) {
                // non-WSI instance (e.g. CEF/ANGLE GPU probe): pure passthrough,
                // no tracking, no injection
                auto res = vkCreateInstance(info, alloc, instance);
                if (res != VK_SUCCESS)
                    throw ls::vulkan_error(res, "vkCreateInstance() failed");
                return VK_SUCCESS;
            }

            VkInstanceCreateInfo newInfo = *info;
            layer_info->root.modifyInstanceCreateInfo(newInfo,
                [=, newInfo = &newInfo]() {
                    auto res = vkCreateInstance(newInfo, alloc, instance);
                    if (res != VK_SUCCESS)
                        throw ls::vulkan_error(res, "vkCreateInstance() failed");
                }
            );

            {
                std::lock_guard<std::mutex> lock(layer_info->mutex);
                layer_info->instanceGipa[*instance] = linkInfo->pfnNextGetInstanceProcAddr;
            }

            if (!instance_info) {
                std::lock_guard<std::mutex> lock(instance_info_mutex);
                if (!instance_info)
                    instance_info = new InstanceInfo{ // NOLINT (memory management)
                        .funcs = vk::initVulkanInstanceFuncs(*instance,
                            linkInfo->pfnNextGetInstanceProcAddr, true),
                    };
            }

            {
                std::lock_guard<std::mutex> lock(instance_info->mutex);
                instance_info->handles.push_back(*instance);
            }

            return VK_SUCCESS;
        } catch (const ls::vulkan_error& e) {
            if (e.error() == VK_ERROR_EXTENSION_NOT_PRESENT)
                std::cerr << "lsfg-vk: required Vulkan instance extensions are not present. "
                    "Your GPU driver is not supported.\n";
            return e.error();
        }
    }

    // create device
    VkResult myvkCreateDevice(
            VkPhysicalDevice physdev,
            const VkDeviceCreateInfo* info,
            const VkAllocationCallbacks* alloc,
            VkDevice* device) {
        // apply layer chaining
        auto* layerInfo = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(info->pNext));
        while (layerInfo && (layerInfo->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
                || layerInfo->function != VK_LAYER_LINK_INFO)) {
            layerInfo = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(layerInfo->pNext));
        }
        if (!layerInfo) {
            std::cerr << "lsfg-vk: no layer info found in pNext chain, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto* linkInfo = layerInfo->u.pLayerInfo;
        if (!linkInfo) {
            std::cerr << "lsfg-vk: link info is null, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        instance_info->funcs.GetDeviceProcAddr = linkInfo->pfnNextGetDeviceProcAddr;
        if (!linkInfo->pfnNextGetDeviceProcAddr) {
            std::cerr << "lsfg-vk: next layer's vkGetDeviceProcAddr is null, "
                "the previous layer does not follow spec\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        layerInfo->u.pLayerInfo = linkInfo->pNext; // advance for next layer

        // fetch device loader functions
        layerInfo = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(info->pNext));
        while (layerInfo && (layerInfo->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO
                || layerInfo->function != VK_LOADER_DATA_CALLBACK)) {
            layerInfo = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(layerInfo->pNext));
        }
        if (!layerInfo) {
            std::cerr << "lsfg-vk: no layer loader data found in pNext chain.\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto* setLoaderData = layerInfo->u.pfnSetDeviceLoaderData;
        if (!setLoaderData) {
            std::cerr << "lsfg-vk: instance loader data function is null.\n";
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        // create device
        try {
            VkDeviceCreateInfo newInfo = *info;
            layer_info->root.modifyDeviceCreateInfo(newInfo,
                [=, newInfo = &newInfo]() {
                    auto res = instance_info->funcs.CreateDevice(physdev, newInfo, alloc, device);
                    if (res != VK_SUCCESS)
                        throw ls::vulkan_error(res, "vkCreateDevice() failed");
                }
            );

            {
                std::lock_guard<std::mutex> lock(instance_info->mutex);
                instance_info->deviceGdpa[*device] = linkInfo->pfnNextGetDeviceProcAddr;
            }
        } catch (const ls::vulkan_error& e) {
            if (e.error() == VK_ERROR_EXTENSION_NOT_PRESENT)
                std::cerr << "lsfg-vk: required Vulkan device extensions are not present. "
                    "Your GPU driver is not supported.\n";
            return e.error();
        }

        // create layer instance
        try {
            instance_info->devices.emplace(
                *device,
                vk::Vulkan(
                    instance_info->handles.front(), *device, physdev,
                    instance_info->funcs, vk::initVulkanDeviceFuncs(instance_info->funcs, *device,
                        true),
                    true, setLoaderData
                )
            );
        } catch (const std::exception& e) {
            std::cerr << "lsfg-vk: something went wrong during lsfg-vk initialization:\n";
            std::cerr << "- " << e.what() << '\n';
        }

        return VK_SUCCESS;
    }

    // destroy device
    void myvkDestroyDevice(VkDevice device, const VkAllocationCallbacks* alloc) {
        PFN_vkGetDeviceProcAddr nextGdpa{};
        {
            std::lock_guard<std::mutex> lock(instance_info->mutex);
            auto it = instance_info->deviceGdpa.find(device);
            if (it != instance_info->deviceGdpa.end()) {
                nextGdpa = it->second;
                instance_info->deviceGdpa.erase(it);
            }

            auto dit = instance_info->devices.find(device);
            if (dit != instance_info->devices.end())
                instance_info->devices.erase(dit);
        }

        // destroy device
        PFN_vkDestroyDevice vkDestroyDevice{};
        if (nextGdpa)
            vkDestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(
                nextGdpa(device, "vkDestroyDevice"));
        else
            vkDestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(
                instance_info->funcs.GetDeviceProcAddr(device, "vkDestroyDevice"));
        if (!vkDestroyDevice) {
            std::cerr << "lsfg-vk: failed to get next layer's vkDestroyDevice, "
                "the previous layer does not follow spec\n";
            return;
        }

        vkDestroyDevice(device, alloc);
    }

    // destroy instance
    void myvkDestroyInstance(VkInstance instance, const VkAllocationCallbacks* alloc) {
        PFN_vkGetInstanceProcAddr nextGipa{};
        {
            std::lock_guard<std::mutex> lock(layer_info->mutex);
            auto git = layer_info->instanceGipa.find(instance);
            if (git != layer_info->instanceGipa.end()) {
                nextGipa = git->second;
                layer_info->instanceGipa.erase(git);
            }
        }

        {
            std::lock_guard<std::mutex> lock(instance_info->mutex);
            // remove instance handle
            auto it = std::ranges::find(instance_info->handles, instance);
            if (it != instance_info->handles.end())
                instance_info->handles.erase(it);

            // destroy instance info if no handles remain
            if (instance_info->handles.empty()) {
                delete instance_info; // NOLINT (memory management)
                instance_info = nullptr;
            }
        }

        // destroy instance
        auto vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
            (nextGipa ? nextGipa : layer_info->GetInstanceProcAddr)(instance, "vkDestroyInstance"));
        if (!vkDestroyInstance) {
            std::cerr << "lsfg-vk: failed to get next layer's vkDestroyInstance, "
                "the previous layer does not follow spec\n";
            return;
        }

        vkDestroyInstance(instance, alloc);
    }

    // get optional function pointer override
    PFN_vkVoidFunction getProcAddr(const std::string& name) {
        auto it = layer_info->map.find(name);
        if (it != layer_info->map.end())
            return it->second;
        return nullptr;
    }

    // get physical-device-level function pointers
    // loader v2 passes the physical device handle in the VkInstance-typed slot
    PFN_vkVoidFunction myvkGetPhysicalDeviceProcAddr(VkInstance instance, const char* name) {
        if (!name) return nullptr;

        auto func = getProcAddr(name);
        if (func) return func;

        if (!layer_info->nextGetPhysicalDeviceProcAddr) { // resolve lazily, once
            std::lock_guard<std::mutex> lock(layer_info->mutex);
            if (!layer_info->nextGetPhysicalDeviceProcAddr) {
                auto p = layer_info->GetInstanceProcAddr(VK_NULL_HANDLE, "vkGetPhysicalDeviceProcAddr");
                layer_info->nextGetPhysicalDeviceProcAddr =
                    reinterpret_cast<PFN_GetPhysicalDeviceProcAddr>(p);
            }
        }

        PFN_vkVoidFunction pfn = nullptr;
        if (layer_info->nextGetPhysicalDeviceProcAddr)
            pfn = layer_info->nextGetPhysicalDeviceProcAddr(instance, name);
        if (!pfn && layer_info->GetInstanceProcAddr)
            pfn = layer_info->GetInstanceProcAddr(VK_NULL_HANDLE, name); // host loader resolves physdev fns via GIPA
        return pfn;
    }

    // get instance-level function pointers
    PFN_vkVoidFunction myvkGetInstanceProcAddr(VkInstance instance, const char* name) {
        if (!name) return nullptr;

        auto func = getProcAddr(name);
        if (func) return func;

        PFN_vkGetInstanceProcAddr gipa = layer_info->GetInstanceProcAddr;
        {
            std::lock_guard<std::mutex> lock(layer_info->mutex);
            auto it = layer_info->instanceGipa.find(instance);
            if (it != layer_info->instanceGipa.end())
                gipa = it->second;
        }
        if (!gipa) return nullptr;

        auto result = gipa(instance, name);
        return result;
    }

    // get device-level function pointers
    PFN_vkVoidFunction myvkGetDeviceProcAddr(VkDevice device, const char* name) {
        if (!name) return nullptr;

        auto func = getProcAddr(name);
        if (func) return func;

        PFN_vkGetDeviceProcAddr gdpa{};
        {
            std::lock_guard<std::mutex> lock(instance_info->mutex);
            auto it = instance_info->deviceGdpa.find(device);
            if (it != instance_info->deviceGdpa.end())
                gdpa = it->second;
        }
        if (!gdpa)
            gdpa = instance_info->funcs.GetDeviceProcAddr;
        if (!gdpa) return nullptr;

        auto result = gdpa(device, name);
        return result;
    }
}

namespace {
    VkResult myvkCreateSwapchainKHR(
            VkDevice device,
            const VkSwapchainCreateInfoKHR* info,
            const VkAllocationCallbacks* alloc,
            VkSwapchainKHR* swapchain) {
        const auto& it = instance_info->devices.find(device);
        if (it == instance_info->devices.end())
            return VK_ERROR_INITIALIZATION_FAILED;

        if (std::getenv("LSFGVK_NO_FG")) {
            // true pass-through: no swapchain modification, no tracking
            auto res = it->second.df().CreateSwapchainKHR(device, info, alloc, swapchain);
            return res;
        }

        try {
            // mark old swapchain as retired
            if (info->oldSwapchain)
                instance_info->retiredSwapchains.emplace(info->oldSwapchain);

            layer_info->root.update(); // ensure config is up to date

            // create swapchain
            VkSwapchainCreateInfoKHR newInfo = *info;
            layer_info->root.modifySwapchainCreateInfo(it->second, newInfo,
                [=, newInfo = &newInfo]() {
                    auto res = it->second.df().CreateSwapchainKHR(
                        device, newInfo, alloc, swapchain);
                    if (res != VK_SUCCESS)
                        throw ls::vulkan_error(res, "vkCreateSwapchainKHR() failed");
                }
            );

            // get all swapchain images
            uint32_t imageCount{};
            auto res = it->second.df().GetSwapchainImagesKHR(device, *swapchain,
                &imageCount, VK_NULL_HANDLE);
            if (res != VK_SUCCESS || imageCount == 0)
                throw ls::vulkan_error(res, "vkGetSwapchainImagesKHR() failed");

            std::vector<VkImage> swapchainImages(imageCount);
            res = it->second.df().GetSwapchainImagesKHR(device, *swapchain,
                &imageCount, swapchainImages.data());
            if (res != VK_SUCCESS)
                throw ls::vulkan_error(res, "vkGetSwapchainImagesKHR() failed");

            auto& swapchainInfo = instance_info->swapchainInfos.emplace(*swapchain, SwapchainInfo {
                .images = std::move(swapchainImages),
                .format = newInfo.imageFormat,
                .colorSpace = newInfo.imageColorSpace,
                .extent = newInfo.imageExtent,
                .presentMode = newInfo.presentMode,
                .surface = newInfo.surface
            }).first->second;

            // create lsfg-vk swapchain
            if (std::getenv("LSFGVK_NO_FG")) {
                return VK_SUCCESS; // leave the swapchain UNTRACKED
            }

            // Degenerate / probe swapchains: frame generation on a near-zero
            // extent is meaningless AND crashes the driver. The 7-level mipmap
            // pyramid (deepest = extent >> 6) needs extent >= 64 to stay
            // non-zero; RADV NULL-derefs in vkBindImageMemory on a 0x0 image.
            // Games create a 1x1 probe swapchain before the real one — pass
            // those through (untracked) so the real swapchain can engage FG.
            {
                const uint32_t minDim = 64;
                if (newInfo.imageExtent.width < minDim || newInfo.imageExtent.height < minDim) {
                    std::cerr << "lsfg-vk: swapchain extent "
                              << newInfo.imageExtent.width << "x" << newInfo.imageExtent.height
                              << " < " << minDim << "x" << minDim
                              << ", skipping FG context (present will forward to driver)\n";
                    return VK_SUCCESS; // leave the swapchain UNTRACKED
                }
            }
            layer_info->root.createSwapchainContext(it->second, *swapchain, swapchainInfo);

            instance_info->swapchains.emplace(*swapchain,
                ls::R<vk::Vulkan>(it->second));

            VkSwapchainKHR replacedSwapchain = info->oldSwapchain;
            if (!replacedSwapchain) {
                for (const auto& [sc, scInfo] : instance_info->swapchainInfos) {
                    if (sc == *swapchain)
                        continue;
                    if (scInfo.surface == swapchainInfo.surface) {
                        replacedSwapchain = sc;
                        break;
                    }
                }
            }

            if (replacedSwapchain) {
                auto pendingIt = instance_info->pendingMultiplier.find(replacedSwapchain);
                if (pendingIt != instance_info->pendingMultiplier.end()) {
                    auto& context = layer_info->root.getSwapchainContext(*swapchain);
                    const size_t capacityMultiplier = context.getCreationMultiplier();
                    const size_t requestedMultiplier = pendingIt->second;
                    auto loggedIt = instance_info->loggedDeferredMultiplier.find(replacedSwapchain);

                    if (requestedMultiplier <= capacityMultiplier) {
                        instance_info->pendingMultiplier.erase(pendingIt);
                        if (loggedIt != instance_info->loggedDeferredMultiplier.end()) {
                            instance_info->loggedDeferredMultiplier[*swapchain] = loggedIt->second;
                            instance_info->loggedDeferredMultiplier.erase(loggedIt);
                        }
                    } else {
                        instance_info->pendingMultiplier[*swapchain] = requestedMultiplier;
                        instance_info->pendingMultiplier.erase(pendingIt);
                        if (loggedIt != instance_info->loggedDeferredMultiplier.end()) {
                            instance_info->loggedDeferredMultiplier[*swapchain] = loggedIt->second;
                            instance_info->loggedDeferredMultiplier.erase(loggedIt);
                        }
                    }
                }
            }

            return res;
        } catch (const ls::vulkan_error& e) {
            std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain creation:\n";
            std::cerr << "- " << e.what() << '\n';
            return e.error();
        } catch (const std::exception& e) {
            std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain creation:\n";
            std::cerr << "- " << e.what() << '\n';
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }

    VkResult myvkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* info) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
        VkResult result = VK_SUCCESS;

        // ensure layer config is up to date
        bool reload{};
        try {
            reload = layer_info->root.update();
        } catch (const std::exception&) {
            reload = false; // ignore parse errors
        }

        if (reload) {
            try {
                if (!layer_info->root.getActiveProfile()) {
                    std::cerr << "lsfg-vk: config reloaded but no active profile, ignoring\n";
                    reload = false;
                }
            } catch (const std::exception&) {
                reload = false;
            }
        }

        std::unordered_map<VkSwapchainKHR, ls::GameConf> effectiveProfiles;
        std::unordered_set<VkSwapchainKHR> forceOutOfDate;
        if (reload && layer_info->root.getActiveProfile()) {
            try {
                const auto& requestedProfile = *layer_info->root.getActiveProfile();

                for (const auto& [swapchain, vk] : instance_info->swapchains) {
                    if (instance_info->retiredSwapchains.find(swapchain)
                            != instance_info->retiredSwapchains.end())
                        continue;

                    auto& context = layer_info->root.getSwapchainContext(swapchain);
                    const size_t capacityMultiplier = context.getCreationMultiplier();
                    const size_t currentMultiplier = context.getProfileMultiplier();
                    const size_t requestedMultiplier = requestedProfile.multiplier;
                    const bool deferAllowed = requestedProfile.defer_multiplier_change;

                    auto profile = requestedProfile;

                    if (profile.reserve_multiplier > capacityMultiplier)
                        profile.reserve_multiplier = capacityMultiplier;

                    if (requestedMultiplier > capacityMultiplier) {
                        if (deferAllowed) {
                            instance_info->pendingMultiplier[swapchain] = requestedMultiplier;
                        } else {
                            instance_info->pendingMultiplier.erase(swapchain);
                            instance_info->loggedDeferredMultiplier.erase(swapchain);
                            forceOutOfDate.emplace(swapchain);
                        }
                        profile.multiplier = currentMultiplier;
                    } else {
                        auto it = instance_info->pendingMultiplier.find(swapchain);
                        if (it != instance_info->pendingMultiplier.end())
                            instance_info->pendingMultiplier.erase(it);
                        profile.multiplier = requestedMultiplier;
                    }

                    effectiveProfiles.emplace(swapchain, profile);
                }
            } catch (const std::exception& e) {
                std::cerr << "lsfg-vk: error checking multiplier: " << e.what() << '\n';
            }
        }

        if (reload && layer_info->root.getActiveProfile()) {
            try {
                for (const auto& [swapchain, vk] : instance_info->swapchains) {
                    if (instance_info->retiredSwapchains.find(swapchain)
                            != instance_info->retiredSwapchains.end())
                        continue;

                    auto profileIt = effectiveProfiles.find(swapchain);
                    if (profileIt == effectiveProfiles.end())
                        continue;

                    auto& swapchainInfo = instance_info->swapchainInfos.at(swapchain);
                    const auto& profile = profileIt->second;

                    layer_info->root.removeSwapchainContext(swapchain);
                    layer_info->root.createSwapchainContext(vk, swapchain, swapchainInfo, profile);
                }
            } catch (const std::exception& e) {
                std::cerr << "lsfg-vk: something went wrong during lsfg-vk configuration update:\n";
                std::cerr << "- " << e.what() << '\n';
            }
        }
        // present each swapchain
        for (size_t i = 0; i < info->swapchainCount; i++) {
            const auto& swapchain = info->pSwapchains[i];

            const auto& it = instance_info->swapchains.find(swapchain);
            if (it == instance_info->swapchains.end()) {
                // untracked swapchain (e.g. LSFGVK_NO_FG): forward to driver
                const vk::Vulkan* dev = nullptr;
                {
                    std::lock_guard<std::mutex> lock(instance_info->mutex);
                    if (!instance_info->devices.empty())
                        dev = &instance_info->devices.begin()->second;
                }
                if (!dev)
                    return VK_ERROR_INITIALIZATION_FAILED;

                VkResult res = dev->df().QueuePresentKHR(
                    queue, const_cast<VkPresentInfoKHR*>(info));
                if (info->pResults)
                    info->pResults[i] = res;
                return res;
            }

            VkResult swapchainResult = VK_SUCCESS;
            bool skipPresent = false;
            bool deferred = false;

            if (instance_info->retiredSwapchains.find(swapchain)
                    != instance_info->retiredSwapchains.end()) {
                swapchainResult = VK_ERROR_OUT_OF_DATE_KHR;
                skipPresent = true;
            }

            if (!skipPresent && forceOutOfDate.find(swapchain) != forceOutOfDate.end()) {
                swapchainResult = VK_ERROR_OUT_OF_DATE_KHR;
                skipPresent = true;
            }

            if (!skipPresent) {
                deferred = instance_info->pendingMultiplier.find(swapchain)
                    != instance_info->pendingMultiplier.end();
            }

            if (!skipPresent) {
                auto& context = layer_info->root.getSwapchainContext(swapchain);
                const size_t currentMultiplier = context.getProfileMultiplier();
                auto pendingIt = instance_info->pendingMultiplier.find(swapchain);
                if (pendingIt != instance_info->pendingMultiplier.end()) {
                    auto loggedIt = instance_info->loggedDeferredMultiplier.find(swapchain);
                    if (loggedIt == instance_info->loggedDeferredMultiplier.end()
                            || loggedIt->second != pendingIt->second) {
                        std::cerr << "lsfg-vk: swapchain " << swapchain
                                  << " multiplier change deferred ("
                                  << currentMultiplier << " -> "
                                  << pendingIt->second
                                  << "), waiting for swapchain recreation\n";
                        instance_info->loggedDeferredMultiplier[swapchain] = pendingIt->second;
                    }
                } else {
                    auto loggedIt = instance_info->loggedDeferredMultiplier.find(swapchain);
                    if (loggedIt != instance_info->loggedDeferredMultiplier.end()) {
                        std::cerr << "lsfg-vk: swapchain " << swapchain
                                  << " multiplier change applied ("
                                  << loggedIt->second << " -> "
                                  << currentMultiplier << ")\n";
                        instance_info->loggedDeferredMultiplier.erase(loggedIt);
                    }
                }
            }

            if (!skipPresent) {
                try {
                    std::vector<VkSemaphore> waitSemaphores;
                    waitSemaphores.reserve(info->waitSemaphoreCount);

                    for (size_t j = 0; j < info->waitSemaphoreCount; j++)
                        waitSemaphores.push_back(info->pWaitSemaphores[j]);

                    auto& context = layer_info->root.getSwapchainContext(swapchain);
                    swapchainResult = context.present(it->second,
                        queue, swapchain,
                        const_cast<void*>(info->pNext),
                        info->pImageIndices[i],
                        { waitSemaphores.begin(), waitSemaphores.end() }
                    );
                } catch (const ls::vulkan_error& e) {
                    if (e.error() != VK_ERROR_OUT_OF_DATE_KHR) {
                        std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain presentation:\n";
                        std::cerr << "- " << e.what() << '\n';
                    }

                    swapchainResult = e.error();
                } catch (const std::exception& e) {
                    std::cerr << "lsfg-vk: something went wrong during lsfg-vk swapchain presentation:\n";
                    std::cerr << "- " << e.what() << '\n';
                    swapchainResult = VK_ERROR_UNKNOWN;
                }
            }

            // NOTE: When a multiplier change is deferred, we still present using the
            // existing swapchain/context but return VK_ERROR_OUT_OF_DATE_KHR to encourage
            // the application to recreate the swapchain. This is intentional to avoid
            // freezing while still nudging proper recreation.
            if (deferred && (swapchainResult == VK_SUCCESS
                    || swapchainResult == VK_SUBOPTIMAL_KHR)) {
                swapchainResult = VK_ERROR_OUT_OF_DATE_KHR;
            }

            if (info->pResults)
                info->pResults[i] = swapchainResult;

            if (swapchainResult == VK_ERROR_OUT_OF_DATE_KHR) {
                result = swapchainResult;
            } else if (result != VK_ERROR_OUT_OF_DATE_KHR &&
                       swapchainResult != VK_SUCCESS) {
                result = swapchainResult;
            } else if (result == VK_SUCCESS) {
                result = swapchainResult;
            }
        }

        return result;
#pragma clang diagnostic pop
    }

    void myvkDestroySwapchainKHR(
            VkDevice device,
            VkSwapchainKHR swapchain,
            const VkAllocationCallbacks* alloc) {
        const auto& it = instance_info->devices.find(device);
        if (it == instance_info->devices.end())
            return;

        // untracked swapchain (e.g. LSFGVK_NO_FG): forward to driver
        if (instance_info->swapchains.find(swapchain) == instance_info->swapchains.end()) {
            it->second.df().DestroySwapchainKHR(device, swapchain, alloc);
            return;
        }

        instance_info->retiredSwapchains.erase(swapchain);
        instance_info->pendingMultiplier.erase(swapchain);
        instance_info->loggedDeferredMultiplier.erase(swapchain);

        const auto& info_mapping = instance_info->swapchainInfos.find(swapchain);
        if (info_mapping != instance_info->swapchainInfos.end())
            instance_info->swapchainInfos.erase(info_mapping);

        const auto& mapping = instance_info->swapchains.find(swapchain);
        if (mapping != instance_info->swapchains.end())
            instance_info->swapchains.erase(mapping);

        layer_info->root.removeSwapchainContext(swapchain);

        // destroy swapchain
        it->second.df().DestroySwapchainKHR(device, swapchain, alloc);
    }
}

/// Vulkan layer entrypoint
__attribute__((visibility("default")))
VkResult vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct) {
    init_file_logging();
    // ensure loader compatibility
    if (!pVersionStruct
        || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT
        || pVersionStruct->loaderLayerInterfaceVersion < 2)
        return VK_ERROR_INITIALIZATION_FAILED;

    // if the layer has already been initialized, skip
    if (layer_info) {
        pVersionStruct->loaderLayerInterfaceVersion = 2;
        pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
        pVersionStruct->pfnGetDeviceProcAddr = myvkGetDeviceProcAddr;
        pVersionStruct->pfnGetInstanceProcAddr = myvkGetInstanceProcAddr;
        return VK_SUCCESS;
    }

    // load the layer configuration
    try {
        layer_info = new LayerInfo { // NOLINT (memory management)
            .map = {
#define VKPTR(name) reinterpret_cast<PFN_vkVoidFunction>(name)
                { "vkCreateInstance", VKPTR(myvkCreateInstance) },
                { "vkCreateDevice", VKPTR(myvkCreateDevice) },
                { "vkDestroyDevice", VKPTR(myvkDestroyDevice) },
                { "vkDestroyInstance", VKPTR(myvkDestroyInstance) },
                { "vkCreateSwapchainKHR", VKPTR(myvkCreateSwapchainKHR) },
                { "vkQueuePresentKHR", VKPTR(myvkQueuePresentKHR) },
                { "vkDestroySwapchainKHR", VKPTR(myvkDestroySwapchainKHR) }
#undef VKPTR
            },
            .root = Root()
        };

        if (!layer_info->root.active()) { // skip inactive
            delete layer_info; // NOLINT (memory management)
            layer_info = nullptr;

            return VK_ERROR_INITIALIZATION_FAILED;
        }
    } catch (const std::exception& e) {
        std::cerr << "lsfg-vk: something went wrong during lsfg-vk layer initialization:\n";
        std::cerr << "- " << e.what() << '\n';

        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // emplace function pointers/version
    pVersionStruct->loaderLayerInterfaceVersion = 2;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    pVersionStruct->pfnGetDeviceProcAddr = myvkGetDeviceProcAddr;
    pVersionStruct->pfnGetInstanceProcAddr = myvkGetInstanceProcAddr;
    return VK_SUCCESS;
}
