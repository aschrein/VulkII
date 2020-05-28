#define UTILS_IMPL
#include "utils.hpp"

//#include <imgui.h>
//#include <imgui/examples/imgui_impl_sdl.h>
//#include <imgui/examples/imgui_impl_vulkan.h>


#ifdef __linux__
#define VK_USE_PLATFORM_XCB_KHR
#include <vulkan/vulkan.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#else
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <SDL.h>
#include <SDL_vulkan.h>
#endif

#define VK_ASSERT_OK(x)                                                                            \
  do {                                                                                             \
    VkResult __res = x;                                                                            \
    if (__res != VK_SUCCESS) {                                                                     \
      fprintf(stderr, "VkResult: %i\n", (i32)__res);                                               \
      TRAP;                                                                                        \
    }                                                                                              \
  } while (0)

struct Window {
	static constexpr u32 MAX_SC_IMAGES = 0x10;
	SDL_Window *         window = 0;

	VkSurfaceKHR surface = VK_NULL_HANDLE;
	i32          window_width = 1280;
	i32          window_height = 720;

	VkInstance       instance = VK_NULL_HANDLE;
	VkPhysicalDevice physdevice = VK_NULL_HANDLE;
	VkQueue          queue = VK_NULL_HANDLE;
	VkDevice         device = VK_NULL_HANDLE;
	VkCommandPool    cmd_pool = VK_NULL_HANDLE;
	VkCommandBuffer  cmd_buffers[MAX_SC_IMAGES] = {};

	VkSwapchainKHR     swapchain = VK_NULL_HANDLE;
	VkRenderPass       sc_render_pass = VK_NULL_HANDLE;
	uint32_t           sc_image_count = 0;
	VkImageLayout      sc_image_layout[MAX_SC_IMAGES] = {};
	VkImage            sc_images[MAX_SC_IMAGES] = {};
	VkImageView        sc_image_views[MAX_SC_IMAGES] = {};
	VkFramebuffer      sc_framebuffers[MAX_SC_IMAGES] = {};
	VkExtent2D         sc_extent = {};
	VkSurfaceFormatKHR sc_format = {};

	u32         frame_id = 0;
	VkFence     frame_fences[MAX_SC_IMAGES] = {};
	VkSemaphore sc_free_sem[MAX_SC_IMAGES] = {};
	VkSemaphore render_finish_sem[MAX_SC_IMAGES] = {};

	u32 graphics_queue_id = 0;
	u32 compute_queue_id = 0;
	u32 transfer_queue_id = 0;

	void release() {
		vkDeviceWaitIdle(device);
		ito(sc_image_count) vkDestroySemaphore(device, sc_free_sem[i], NULL);
		ito(sc_image_count) vkDestroySemaphore(device, render_finish_sem[i], NULL);
		ito(sc_image_count) vkDestroyFence(device, frame_fences[i], NULL);
		vkDestroySwapchainKHR(device, swapchain, NULL);
		vkDestroyDevice(device, NULL);
		vkDestroySurfaceKHR(instance, surface, NULL);
		vkDestroyInstance(instance, NULL);
		SDL_DestroyWindow(window);
	}
	void release_swapchain() {
		if (swapchain != VK_NULL_HANDLE) {
			vkDestroySwapchainKHR(device, swapchain, NULL);
		}
		ito(sc_image_count) {
			if (sc_framebuffers[i] != VK_NULL_HANDLE)
				vkDestroyFramebuffer(device, sc_framebuffers[i], NULL);
			if (sc_image_views[i] != VK_NULL_HANDLE) vkDestroyImageView(device, sc_image_views[i], NULL);
		}
		if (sc_render_pass != VK_NULL_HANDLE) {
			vkDestroyRenderPass(device, sc_render_pass, NULL);
		}
	}

	void update_swapchain() {
		SDL_SetWindowResizable(window, SDL_FALSE);
		defer(SDL_SetWindowResizable(window, SDL_TRUE));
		vkDeviceWaitIdle(device);
		release_swapchain();
		u32                format_count = 0;
		VkSurfaceFormatKHR formats[0x100];
		vkGetPhysicalDeviceSurfaceFormatsKHR(physdevice, surface, &format_count, 0);
		vkGetPhysicalDeviceSurfaceFormatsKHR(physdevice, surface, &format_count, formats);
		VkSurfaceFormatKHR format_of_choice;
		format_of_choice.format = VK_FORMAT_UNDEFINED;
		ito(format_count) {
			if (formats[i].format == VK_FORMAT_R8G8B8A8_SRGB ||  //
				formats[i].format == VK_FORMAT_B8G8R8A8_SRGB ||  //
				formats[i].format == VK_FORMAT_B8G8R8_SRGB ||    //
				formats[i].format == VK_FORMAT_R8G8B8_SRGB ||    //
				formats[i].format == VK_FORMAT_R8G8B8_UNORM ||   //
				formats[i].format == VK_FORMAT_R8G8B8A8_UNORM || //
				formats[i].format == VK_FORMAT_B8G8R8A8_UNORM || //
				formats[i].format == VK_FORMAT_B8G8R8_UNORM      //
				) {
				format_of_choice = formats[i];
				break;
			}
		}
		ASSERT_ALWAYS(format_of_choice.format != VK_FORMAT_UNDEFINED);
		sc_format = format_of_choice;

		uint32_t num_present_modes = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(physdevice, surface, &num_present_modes, NULL);
		VkPresentModeKHR present_modes[0x100];
		vkGetPhysicalDeviceSurfacePresentModesKHR(physdevice, surface, &num_present_modes,
			present_modes);
		VkPresentModeKHR present_mode_of_choice = VK_PRESENT_MODE_FIFO_KHR; // always supported.
		ito(num_present_modes) {
			if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) { // prefer mailbox
				present_mode_of_choice = VK_PRESENT_MODE_MAILBOX_KHR;
				break;
			}
		}
		//    usleep(100000);
		VkSurfaceCapabilitiesKHR surface_capabilities;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physdevice, surface, &surface_capabilities);

		sc_extent = surface_capabilities.currentExtent;
		sc_extent.width = CLAMP(sc_extent.width, surface_capabilities.minImageExtent.width,
			surface_capabilities.maxImageExtent.width);
		sc_extent.height = CLAMP(sc_extent.height, surface_capabilities.minImageExtent.height,
			surface_capabilities.maxImageExtent.height);

		VkSwapchainCreateInfoKHR sc_create_info;
		MEMZERO(sc_create_info);
		sc_create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		sc_create_info.surface = surface;
		sc_create_info.minImageCount = CLAMP(3, surface_capabilities.minImageCount, surface_capabilities.maxImageCount);
		sc_create_info.imageFormat = format_of_choice.format;
		sc_create_info.imageColorSpace = format_of_choice.colorSpace;
		sc_create_info.imageExtent = sc_extent;
		sc_create_info.imageArrayLayers = 1;
		sc_create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		sc_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		sc_create_info.preTransform = surface_capabilities.currentTransform;
		sc_create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		sc_create_info.presentMode = present_mode_of_choice;
		sc_create_info.clipped = VK_TRUE;
		sc_image_count = 0;
		VK_ASSERT_OK(vkCreateSwapchainKHR(device, &sc_create_info, 0, &swapchain));
		vkGetSwapchainImagesKHR(device, swapchain, &sc_image_count, NULL);
		vkGetSwapchainImagesKHR(device, swapchain, &sc_image_count, sc_images);
		ito(sc_image_count) {
			VkImageViewCreateInfo view_ci;
			MEMZERO(view_ci);
			view_ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			view_ci.image = sc_images[i];
			MEMZERO(view_ci.components);
			view_ci.components.r = VK_COMPONENT_SWIZZLE_R;
			view_ci.components.g = VK_COMPONENT_SWIZZLE_G;
			view_ci.components.b = VK_COMPONENT_SWIZZLE_B;
			view_ci.components.a = VK_COMPONENT_SWIZZLE_A;
			view_ci.format = sc_format.format;
			MEMZERO(view_ci.subresourceRange);
			view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			view_ci.subresourceRange.baseMipLevel = 0;
			view_ci.subresourceRange.levelCount = 1;
			view_ci.subresourceRange.baseArrayLayer = 0;
			view_ci.subresourceRange.layerCount = 1;
			view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
			VK_ASSERT_OK(vkCreateImageView(device, &view_ci, NULL, &sc_image_views[i]));
			sc_image_layout[i] = VK_IMAGE_LAYOUT_UNDEFINED;
		}
		{
			VkAttachmentDescription attachment;
			MEMZERO(attachment);
			attachment.format = sc_format.format;
			attachment.samples = VK_SAMPLE_COUNT_1_BIT;
			attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
			attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			VkAttachmentReference color_attachment;
			MEMZERO(color_attachment);
			color_attachment.attachment = 0;
			color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			VkSubpassDescription subpass;
			MEMZERO(subpass);
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = 1;
			subpass.pColorAttachments = &color_attachment;
			VkSubpassDependency dependency;
			MEMZERO(dependency);
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = 0;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			VkRenderPassCreateInfo info;
			MEMZERO(info);
			info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			info.attachmentCount = 1;
			info.pAttachments = &attachment;
			info.subpassCount = 1;
			info.pSubpasses = &subpass;
			info.dependencyCount = 1;
			info.pDependencies = &dependency;
			VK_ASSERT_OK(vkCreateRenderPass(device, &info, NULL, &sc_render_pass));
		}
		ito(sc_image_count) {
			VkFramebufferCreateInfo info;
			MEMZERO(info);
			info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			info.attachmentCount = 1;
			info.width = sc_extent.width;
			info.height = sc_extent.height;
			info.layers = 1;
			info.pAttachments = &sc_image_views[i];
			info.renderPass = sc_render_pass;
			VK_ASSERT_OK(vkCreateFramebuffer(device, &info, NULL, &sc_framebuffers[i]));
		}
	}

	void init() {
		SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
		SDL_Window *window = SDL_CreateWindow("VulkII", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
			1280, 720, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
		TMP_STORAGE_SCOPE;

		u32 num_instance_extensions;
		ASSERT_ALWAYS(SDL_Vulkan_GetInstanceExtensions(window, &num_instance_extensions, nullptr));
		const char **instance_extensions =
			(char const **)tl_alloc_tmp((num_instance_extensions + 1) * sizeof(char *));
		ASSERT_ALWAYS(
			SDL_Vulkan_GetInstanceExtensions(window, &num_instance_extensions, instance_extensions));
		instance_extensions[num_instance_extensions++] = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;

		VkApplicationInfo app_info;
		MEMZERO(app_info);
		app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app_info.apiVersion = VK_API_VERSION_1_2;
		app_info.applicationVersion = 1;
		app_info.pApplicationName = "Vulkii";
		app_info.pEngineName = "Vulkii";

		const char *layerNames[] = {//
																// "VK_LAYER_LUNARG_standard_validation" // [Deprecated]
																"VK_LAYER_KHRONOS_validation", NULL };

		VkInstanceCreateInfo info;
		MEMZERO(info);
		info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		info.pApplicationInfo = &app_info;
		info.enabledLayerCount = ARRAY_SIZE(layerNames) - 1;
		info.ppEnabledLayerNames = layerNames;
		info.enabledExtensionCount = num_instance_extensions;
		info.ppEnabledExtensionNames = instance_extensions;

		VK_ASSERT_OK(vkCreateInstance(&info, nullptr, &instance));

		if (!SDL_Vulkan_CreateSurface(window, instance, &surface)) {
			TRAP;
		}
		const u32               MAX_COUNT = 0x100;
		u32                     physdevice_count;
		VkPhysicalDevice        physdevice_handles[MAX_COUNT];
		VkQueueFamilyProperties queue_family_properties[MAX_COUNT];
		//  VkQueueFamilyProperties2    queue_family_properties2[MAX_COUNT];
		//  VkQueueFamilyProperties2KHR queue_family_properties2KHR[MAX_COUNT];

		vkEnumeratePhysicalDevices(instance, &physdevice_count, 0);
		vkEnumeratePhysicalDevices(instance, &physdevice_count, physdevice_handles);

		VkPhysicalDevice graphics_device_id = NULL;

		ito(physdevice_count) {
			{
				u32 num_queue_family_properties = 0;
				vkGetPhysicalDeviceQueueFamilyProperties(physdevice_handles[i],
					&num_queue_family_properties, NULL);
				vkGetPhysicalDeviceQueueFamilyProperties(
					physdevice_handles[i], &num_queue_family_properties, queue_family_properties);

				jto(num_queue_family_properties) {

					VkBool32 sup = VK_FALSE;
					vkGetPhysicalDeviceSurfaceSupportKHR(physdevice_handles[i], j, surface, &sup);

					if (sup && (queue_family_properties[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
						graphics_queue_id = j;
						graphics_device_id = physdevice_handles[i];
					}
					if (sup && (queue_family_properties[j].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
						compute_queue_id = j;
					}
					if (sup && (queue_family_properties[j].queueFlags & VK_QUEUE_TRANSFER_BIT)) {
						transfer_queue_id = j;
					}
				}
			}
			//    {
			//      u32 num_queue_family_properties = 0;
			//      vkGetPhysicalDeviceQueueFamilyProperties2(physdevice_handles[i],
			//      &num_queue_family_properties,
			//                                                NULL);
			//      vkGetPhysicalDeviceQueueFamilyProperties2(physdevice_handles[i],
			//      &num_queue_family_properties,
			//                                                queue_family_properties2);
			//    }
			//    {
			//      u32 num_queue_family_properties = 0;
			//      vkGetPhysicalDeviceQueueFamilyProperties2KHR(physdevice_handles[i],
			//                                                   &num_queue_family_properties, NULL);
			//      vkGetPhysicalDeviceQueueFamilyProperties2KHR(
			//          physdevice_handles[i], &num_queue_family_properties, queue_family_properties2KHR);
			//    }
		}
		char const *            device_extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
		VkDeviceCreateInfo      deviceCreateInfo;
		MEMZERO(deviceCreateInfo);
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = 0;
		deviceCreateInfo.enabledLayerCount = 0;
		deviceCreateInfo.ppEnabledLayerNames = 0;
		deviceCreateInfo.enabledExtensionCount = ARRAY_SIZE(device_extensions);
		deviceCreateInfo.ppEnabledExtensionNames = device_extensions;
		deviceCreateInfo.pEnabledFeatures = 0;
		float                   priority = 1.0f;
		VkDeviceQueueCreateInfo queue_create_info;
		MEMZERO(queue_create_info);
		queue_create_info.sType =
			VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_create_info.queueFamilyIndex = graphics_queue_id;
		queue_create_info.queueCount = 1;
		queue_create_info.pQueuePriorities = &priority;
		deviceCreateInfo.pQueueCreateInfos = &queue_create_info;
		VK_ASSERT_OK(vkCreateDevice(graphics_device_id, &deviceCreateInfo, NULL, &device));
		vkGetDeviceQueue(device, graphics_queue_id, 0, &queue);
		ASSERT_ALWAYS(queue != VK_NULL_HANDLE);
		physdevice = graphics_device_id;
		update_swapchain();
		{
			VkCommandPoolCreateInfo info;
			MEMZERO(info);
			info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			info.queueFamilyIndex = graphics_queue_id;

			vkCreateCommandPool(device, &info, 0, &cmd_pool);
		}
		{
			VkCommandBufferAllocateInfo info;
			MEMZERO(info);
			info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			info.commandPool = cmd_pool;
			info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			info.commandBufferCount = sc_image_count;

			vkAllocateCommandBuffers(device, &info, cmd_buffers);
		}
		{
			VkSemaphoreCreateInfo info;
			MEMZERO(info);
			info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO; info.pNext = NULL; info.flags = 0;
			ito(sc_image_count) vkCreateSemaphore(device, &info, 0, &sc_free_sem[i]);
			ito(sc_image_count) vkCreateSemaphore(device, &info, 0, &render_finish_sem[i]);
		}
		{
			VkFenceCreateInfo info;
			MEMZERO(info);
			info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
			ito(sc_image_count) vkCreateFence(device, &info, 0, &frame_fences[i]);
		}
	}

	void update_surface_size() {
		VkSurfaceCapabilitiesKHR surface_capabilities;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physdevice, surface, &surface_capabilities);
		window_width = surface_capabilities.currentExtent.width;
		window_height = surface_capabilities.currentExtent.height;
	}
	void start_frame() {
	restart:
		update_surface_size();
		if (window_width != (i32)sc_extent.width || window_height != (i32)sc_extent.height) {
			update_swapchain();
		}

		uint32_t cmd_index = (frame_id++) % sc_image_count;
		VkResult wait_res = vkWaitForFences(device, 1, &frame_fences[cmd_index], VK_TRUE, 1000);
		if (wait_res == VK_TIMEOUT) {
			goto restart;
		}
		vkResetFences(device, 1, &frame_fences[cmd_index]);

		uint32_t image_index;
		VkResult acquire_res = vkAcquireNextImageKHR(
			device, swapchain, UINT64_MAX, sc_free_sem[cmd_index], VK_NULL_HANDLE, &image_index);

		if (acquire_res == VK_ERROR_OUT_OF_DATE_KHR || acquire_res == VK_SUBOPTIMAL_KHR) {
			update_swapchain();
			goto restart;
		}
		else if (acquire_res != VK_SUCCESS) {
			TRAP;
		}

		VkCommandBufferBeginInfo begin_info;
		MEMZERO(begin_info);
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkResetCommandBuffer(cmd_buffers[cmd_index], VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
		vkBeginCommandBuffer(cmd_buffers[cmd_index], &begin_info);
		VkImageSubresourceRange srange;
		MEMZERO(srange);
		srange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		srange.baseMipLevel = 0;
		srange.levelCount = VK_REMAINING_MIP_LEVELS;
		srange.baseArrayLayer = 0;
		srange.layerCount = VK_REMAINING_ARRAY_LAYERS;
		{
			VkImageMemoryBarrier bar;
			MEMZERO(bar);
			bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			bar.srcAccessMask = 0;
			bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			bar.oldLayout = sc_image_layout[image_index];
			bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			bar.srcQueueFamilyIndex = graphics_queue_id;
			bar.dstQueueFamilyIndex = graphics_queue_id;
			bar.image = sc_images[image_index];
			bar.subresourceRange = srange;
			vkCmdPipelineBarrier(cmd_buffers[cmd_index], VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &bar);
		}
		VkClearColorValue clear_color = { {1.0f, 0.0f, 0.0f, 1.0f} };
		vkCmdClearColorImage(cmd_buffers[cmd_index], sc_images[image_index],
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear_color, 1, &srange);
		{
			VkImageMemoryBarrier bar;
			MEMZERO(bar);
			bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			bar.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
			bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			bar.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			bar.srcQueueFamilyIndex = graphics_queue_id;
			bar.dstQueueFamilyIndex = graphics_queue_id;
			bar.image = sc_images[image_index];
			bar.subresourceRange = srange;
			vkCmdPipelineBarrier(cmd_buffers[cmd_index], VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &bar);
		}
		sc_image_layout[image_index] = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		vkEndCommandBuffer(cmd_buffers[cmd_index]);
		VkPipelineStageFlags stage_flags[]{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSubmitInfo         submit_info;
		MEMZERO(submit_info);
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = &sc_free_sem[cmd_index];
		submit_info.pWaitDstStageMask = stage_flags;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &cmd_buffers[cmd_index];
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = &render_finish_sem[cmd_index];
		vkQueueSubmit(queue, 1, &submit_info, frame_fences[cmd_index]);
		VkPresentInfoKHR present_info;
		MEMZERO(present_info);
		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores = &render_finish_sem[cmd_index];
		present_info.swapchainCount = 1;
		present_info.pSwapchains = &swapchain;
		present_info.pImageIndices = &image_index;
		vkQueuePresentKHR(queue, &present_info);
	}
};

int main(int argc, char *argv[]) {
	(void)argc;
	(void)argv;
	Window wnd;
	wnd.init();
	while (true) {
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) {
				exit(0);
			}
			switch (event.type) {
			case SDL_WINDOWEVENT:
				if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
					// Sleep(100000);
				}
				break;
			}

		}
		wnd.start_frame();
	}
	wnd.release();

	SDL_Quit();

	return 0;
}
