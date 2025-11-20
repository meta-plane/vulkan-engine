#ifndef VULKAN_APP_H
#define VULKAN_APP_H

#include <vulkan/vulkan_core.h>
#include <vector>
#include <map>
#include <string>
#include <variant>
#include <optional>
#include <memory>
#include <tuple>
#include <utility> 
#define VULKAN_VERSION_1_3  // TODO: whether to use this or not depends on the system


extern PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR_;
extern PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_;
extern PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_;
extern PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR_;
extern PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR_;
extern PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR_;
extern PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR_;
extern PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR_;
extern PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR_;


namespace ve {

class VulkanApp;
class Device;
class Queue;
class CommandPool;
class CommandBuffer;
class Fence;
class Semaphore;

class ShaderModule;
class ComputePipeline;
class GraphicsPipeline;
class RaytracingPipeline;

class Buffer;
class Image;
class ImageView;
class Sampler;

class DescriptorSetLayout;
class PipelineLayout;
class DescriptorPool;
class DescriptorSet;

class Window;
class AccelerationStructure;


#define VULKAN_FRIENDS \
    friend class VulkanApp; \
    friend class Device; \
    friend class Queue; \
    friend class CommandPool; \
    friend class CommandBuffer; \
    friend class Fence; \
    friend class Semaphore; \
    friend class ShaderModule; \
    friend class ComputePipeline; \
    friend class GraphicsPipeline; \
    friend class RaytracingPipeline; \
    friend class Buffer; \
    friend class Image; \
    friend class ImageView; \
    friend class DescriptorSetLayout; \
    friend class PipelineLayout; \
    friend class DescriptorPool; \
    friend class DescriptorSet; \
    friend class Window; \
    friend class AccelerationStructure; \
    friend class Submitting;


#define VULKAN_CLASS_COMMON \
    VULKAN_FRIENDS \
    struct Impl; Impl* pImpl; \
    Impl& impl() { return *pImpl; } \
    const Impl& impl() const { return *pImpl; } \


#define VULKAN_CLASS_COMMON2(class_name) \
    VULKAN_FRIENDS \
    struct Impl; \
    Impl** ppImpl; \
public: \
    class_name(class_name::Impl** ppImpl=nullptr) : ppImpl(ppImpl) {} \
    operator bool() const { return ppImpl && *ppImpl; } \
    bool operator==(const class_name&) const = default; \
    void destroy(); \
private: \
    Impl& impl() { return **ppImpl; } \
    const Impl& impl() const { return **ppImpl; } \



struct ShaderModuleCreateInfo;
struct ComputePipelineCreateInfo;
struct RaytracingPipelineCreateInfo;
struct BufferCreateInfo;
struct ImageCreateInfo;
struct ImageViewDesc;
struct SamplerCreateInfo;
struct DescriptorPoolCreateInfo;
struct BufferRange;
struct SemaphoreStage;
struct QueueSelector;
struct BindingInfo;
struct DescriptorSetLayoutDesc;
struct PipelineLayoutDesc;

struct MemoryBarrier;
struct BufferMemoryBarrier;
struct ImageMemoryBarrier;

struct CopyRegion;

struct WindowCreateInfo;
struct AsCreateInfo;
// struct AsSizeQueryInfo;
// struct AsGeometryInfo;
struct AsBuildInfoTriangles;
struct AsBuildInfoAabbs;
struct AsBuildInfoInstances;
using AsBuildInfo22 = std::variant<
    AsBuildInfoTriangles,
    AsBuildInfoAabbs,
    AsBuildInfoInstances
>;
struct AsBuildInfo;
struct ShaderBindingTable;


using BarrierInfo = std::variant<MemoryBarrier, BufferMemoryBarrier, ImageMemoryBarrier>;
using Pipeline = std::variant<ComputePipeline, GraphicsPipeline, RaytracingPipeline>;
using Resource = std::variant<Buffer, Image>;
using SubmissionBatchInfo = std::tuple<
    std::vector<SemaphoreStage>, 
    std::vector<CommandBuffer>, 
    std::vector<SemaphoreStage>
>; 


struct BufferDescriptor;
struct ImageDescriptor;
using Descriptor = std::variant<BufferDescriptor, ImageDescriptor, AccelerationStructure>;



enum class SHADER_STAGE : uint32_t {
    NONE                      = 0x00000000,
    VERTEX                    = 0x00000001, 
    TESSELLATION_CONTROL      = 0x00000002, 
    TESSELLATION_EVALUATION   = 0x00000004, 
    GEOMETRY                  = 0x00000008, 
    FRAGMENT                  = 0x00000010, 
    ALL_GRAPHICS              = 0x0000001F,
    COMPUTE                   = 0x00000020, 
    TASK                      = 0x00000040, 
    MESH                      = 0x00000080, 
    RAYGEN                    = 0x00000100, 
    ANY_HIT                   = 0x00000200, 
    CLOSEST_HIT               = 0x00000400, 
    MISS                      = 0x00000800, 
    INTERSECTION              = 0x00001000, 
    CALLABLE                  = 0x00002000, 
    ALL                       = 0x7FFFFFFF,
};
inline constexpr SHADER_STAGE operator|(SHADER_STAGE lhs, SHADER_STAGE rhs)         { return (SHADER_STAGE) ((uint32_t)lhs | (uint32_t)rhs); }
inline SHADER_STAGE& operator|=(SHADER_STAGE& lhs, SHADER_STAGE rhs)                { lhs = lhs | rhs; return lhs; }


enum class DESCRIPTOR_TYPE : uint32_t {
    SAMPLER                                 = 0,
    COMBINED_IMAGE_SAMPLER                  = 1,
    SAMPLED_IMAGE                           = 2,
    STORAGE_IMAGE                           = 3,
    UNIFORM_TEXEL_BUFFER                    = 4,
    STORAGE_TEXEL_BUFFER                    = 5,
    UNIFORM_BUFFER                          = 6,
    STORAGE_BUFFER                          = 7,
    UNIFORM_BUFFER_DYNAMIC                  = 8,
    STORAGE_BUFFER_DYNAMIC                  = 9,
    INPUT_ATTACHMENT                        = 10,
    INLINE_UNIFORM_BLOCK                    = 1000138000,
    ACCELERATION_STRUCTURE_KHR              = 1000150000,
    MUTABLE_EXT                             = 1000351000,
    SAMPLE_WEIGHT_IMAGE_QCOM                = 1000440000,
    BLOCK_MATCH_IMAGE_QCOM                  = 1000440001,
    PARTITIONED_ACCELERATION_STRUCTURE_NV   = 1000570000,
    MAX_ENUM                                = 0x7FFFFFFF,
};


enum class PIPELINE_STAGE : uint64_t {
    NONE                              =              0ULL,
    TOP_OF_PIPE                       =     0x00000001ULL,
    DRAW_INDIRECT                     =     0x00000002ULL,
    VERTEX_INPUT                      =     0x00000004ULL,
    VERTEX_SHADER                     =     0x00000008ULL,
    TESSELLATION_CONTROL_SHADER       =     0x00000010ULL,
    TESSELLATION_EVALUATION_SHADER    =     0x00000020ULL,
    GEOMETRY_SHADER                   =     0x00000040ULL,
    FRAGMENT_SHADER                   =     0x00000080ULL,
    EARLY_FRAGMENT_TESTS              =     0x00000100ULL,
    LATE_FRAGMENT_TESTS               =     0x00000200ULL,
    COLOR_ATTACHMENT_OUTPUT           =     0x00000400ULL,
    COMPUTE_SHADER                    =     0x00000800ULL,
    TRANSFER                          =     0x00001000ULL,
    BOTTOM_OF_PIPE                    =     0x00002000ULL,
    HOST                              =     0x00004000ULL,
    ALL_GRAPHICS                      =     0x00008000ULL,
    ALL_COMMANDS                      =     0x00010000ULL,
    COMMAND_PREPROCESS                =     0x00020000ULL,
    CONDITIONAL_RENDERING             =     0x00040000ULL,
    TASK_SHADER                       =     0x00080000ULL,
    MESH_SHADER                       =     0x00100000ULL,
    RAY_TRACING_SHADER                =     0x00200000ULL,
    FRAGMENT_SHADING_RATE_ATTACHMENT  =     0x00400000ULL,
    FRAGMENT_DENSITY_PROCESS          =     0x00800000ULL,
    TRANSFORM_FEEDBACK                =     0x01000000ULL,
    ACCELERATION_STRUCTURE_BUILD      =     0x02000000ULL,
#ifdef VULKAN_VERSION_1_3
    VIDEO_DECODE                      =     0x04000000ULL,
    VIDEO_ENCODE                      =     0x08000000ULL,
    ACCELERATION_STRUCTURE_COPY       =     0x10000000ULL,
    OPTICAL_FLOW                      =     0x20000000ULL,
    MICROMAP_BUILD                    =     0x40000000ULL,
    COPY                              =    0x100000000ULL,
    RESOLVE                           =    0x200000000ULL,
    BLIT                              =    0x400000000ULL,
    CLEAR                             =    0x800000000ULL,
    INDEX_INPUT                       =   0x1000000000ULL,
    VERTEX_ATTRIBUTE_INPUT            =   0x2000000000ULL,
    PRE_RASTERIZATION_SHADERS         =   0x4000000000ULL,
    CONVERT_COOPERATIVE_VECTOR_MATRIX = 0x100000000000ULL,
#endif
};
inline constexpr PIPELINE_STAGE operator|(PIPELINE_STAGE lhs, PIPELINE_STAGE rhs)   { return (PIPELINE_STAGE) ((uint64_t)lhs | (uint64_t)rhs); }


enum class ACCESS : uint64_t {
    NONE                                  =              0ULL,
    INDIRECT_COMMAND_READ                 =     0x00000001ULL,
    INDEX_READ                            =     0x00000002ULL,
    VERTEX_ATTRIBUTE_READ                 =     0x00000004ULL,
    UNIFORM_READ                          =     0x00000008ULL,
    INPUT_ATTACHMENT_READ                 =     0x00000010ULL,
    SHADER_READ                           =     0x00000020ULL,
    SHADER_WRITE                          =     0x00000040ULL,
    COLOR_ATTACHMENT_READ                 =     0x00000080ULL,
    COLOR_ATTACHMENT_WRITE                =     0x00000100ULL,
    DEPTH_STENCIL_ATTACHMENT_READ         =     0x00000200ULL,
    DEPTH_STENCIL_ATTACHMENT_WRITE        =     0x00000400ULL,
    TRANSFER_READ                         =     0x00000800ULL,
    TRANSFER_WRITE                        =     0x00001000ULL,
    HOST_READ                             =     0x00002000ULL,
    HOST_WRITE                            =     0x00004000ULL,
    MEMORY_READ                           =     0x00008000ULL,
    MEMORY_WRITE                          =     0x00010000ULL,
    COMMAND_PREPROCESS_READ               =     0x00020000ULL,
    COMMAND_PREPROCESS_WRITE              =     0x00040000ULL,
    COLOR_ATTACHMENT_READ_NONCOHERENT     =     0x00080000ULL,
    CONDITIONAL_RENDERING_READ            =     0x00100000ULL,
    ACCELERATION_STRUCTURE_READ           =     0x00200000ULL,
    ACCELERATION_STRUCTURE_WRITE          =     0x00400000ULL,
    FRAGMENT_SHADING_RATE_ATTACHMENT_READ =     0x00800000ULL,
    FRAGMENT_DENSITY_MAP_READ             =     0x01000000ULL,
    TRANSFORM_FEEDBACK_WRITE              =     0x02000000ULL,
    TRANSFORM_FEEDBACK_COUNTER_READ       =     0x04000000ULL,
    TRANSFORM_FEEDBACK_COUNTER_WRITE      =     0x08000000ULL,
#ifdef VULKAN_VERSION_1_3
    SHADER_SAMPLED_READ                   =    0x100000000ULL,
    SHADER_STORAGE_READ                   =    0x200000000ULL,
    SHADER_STORAGE_WRITE                  =    0x400000000ULL,
    VIDEO_DECODE_READ                     =    0x800000000ULL,
    VIDEO_DECODE_WRITE                    =   0x1000000000ULL,
    VIDEO_ENCODE_READ                     =   0x2000000000ULL,
    VIDEO_ENCODE_WRITE                    =   0x4000000000ULL,
    SHADER_BINDING_TABLE_READ             =  0x10000000000ULL,
    DESCRIPTOR_BUFFER_READ                =  0x20000000000ULL,
    OPTICAL_FLOW_READ                     =  0x40000000000ULL,
    OPTICAL_FLOW_WRITE                    =  0x80000000000ULL,
    MICROMAP_READ                         = 0x100000000000ULL,
    MICROMAP_WRITE                        = 0x200000000000ULL,
#endif
};
inline constexpr ACCESS operator|(ACCESS lhs, ACCESS rhs)       { return (ACCESS) ((uint64_t)lhs | (uint64_t)rhs); }



enum class IMAGE_LAYOUT : uint32_t {
    UNDEFINED                 = 0,
    GENERAL                   = 1,
    COLOR_ATTACHMENT          = 2,
    DEPTH_STENCIL_ATTACHMENT  = 3,
    DEPTH_STENCIL_READ_ONLY   = 4,
    SHADER_READ_ONLY          = 5,
    TRANSFER_SRC              = 6,
    TRANSFER_DST              = 7,
    PREINITIALIZED            = 8,
    PRESENT_SRC               = 1000001002,
    MAX_ENUM                  = 0x7FFFFFFF,
};


/*
* From [Table 69. Required Limits] in the spec:
*/
namespace portable {
    constexpr uint32_t minMemoryMapAlignment = 64;      // min, vkMapMemory() minimum alignment
    constexpr uint32_t shaderGroupHandleSize = 32;      // exact, Size of a shader group handle
    constexpr uint32_t shaderGroupBaseAlignment = 64;   // max, Alignment for SBT base addresses
    constexpr uint32_t shaderGroupHandleAlignment = 32; // max, Alignment for SBT record addresses
    constexpr uint32_t maxShaderGroupStride = 4096;     // min, Maximum SBT record size
}


struct ShaderGroupHandle {
    uint8_t data[portable::shaderGroupHandleSize];
};



struct DeviceSettings {
    bool enableGraphicsQueues;
    bool enableComputeQueues;
    bool enableTransferQueues;
    bool enablePresent;
    bool enableRaytracing;
    // bool operator==(const DeviceSettings&) const = default; 
    bool operator<=(const DeviceSettings& other) const {
        return (!enableGraphicsQueues || other.enableGraphicsQueues) &&
               (!enableComputeQueues  || other.enableComputeQueues)  &&
               (!enableTransferQueues || other.enableTransferQueues) &&
               (!enablePresent        || other.enablePresent)        &&
               (!enableRaytracing     || other.enableRaytracing);
    }
};


enum QueueType {
    queue_graphics, 
    queue_compute, 
    queue_transfer, 
    queue_max,
};


enum class OwnershipTransferOpType {
    none,
    release,
    acquire,
};


class VulkanApp {
    VULKAN_CLASS_COMMON
    ~VulkanApp();
    VulkanApp();
    VulkanApp(const VulkanApp&) = delete;
    VulkanApp& operator=(const VulkanApp&) = delete;
    Device createDevice(const DeviceSettings& settings);
    Device createDevice(VkPhysicalDevice pd, const DeviceSettings& settings);
    
public:
    static VulkanApp& get();    // singleton pattern
    uint32_t deviceCount() const;
    Device device(int gpuIndex=-1); 
    Device device(DeviceSettings settings);  

    Window createWindow(WindowCreateInfo info);
    void destroyWindow(Window window);
};


class Device {
    VULKAN_CLASS_COMMON2(Device)

public:
    void reportGPUQueueFamilies() const;
    void reportAssignedQueues() const;    

    uint32_t queueCount(QueueType type) const;
    bool supportPresent(QueueType type) const;
    Queue queue(QueueType type, uint32_t index=0) const;
    QueueSelector queue(uint32_t index=0) const;

    CommandPool createCommandPool(QueueType type, VkCommandPoolCreateFlags flags=0);
    CommandPool setDefalutCommandPool(QueueType type, CommandPool cmdPool);
    CommandBuffer newCommandBuffer(QueueType type, VkCommandPoolCreateFlags poolFlags=0);

    Fence createFence(VkFenceCreateFlags flags=0);
    VkResult waitFences(std::vector<Fence> fences, bool waitAll, uint64_t timeout=uint64_t(-1));
    void resetFences(std::vector<Fence> fences);
    Semaphore createSemaphore();
    template <int N> auto createSemaphores()
    {
        return [this]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(((void)I, createSemaphore())...);
        }(std::make_index_sequence<N>{});
    }

    ShaderModule createShaderModule(const ShaderModuleCreateInfo& info);
    ComputePipeline createComputePipeline(const ComputePipelineCreateInfo& info);
    RaytracingPipeline createRaytracingPipeline(const RaytracingPipelineCreateInfo& info);
    Buffer createBuffer(const BufferCreateInfo& info) ;
    Image createImage(const ImageCreateInfo& info);
    Sampler createSampler(const SamplerCreateInfo& info);
    DescriptorSetLayout createDescriptorSetLayout(DescriptorSetLayoutDesc desc); // call-by-value is ok because at least one copy is necessary for lvalue
    PipelineLayout createPipelineLayout(PipelineLayoutDesc desc);
    DescriptorPool createDescriptorPool(const DescriptorPoolCreateInfo& info);

    // VkAccelerationStructureBuildSizesInfoKHR getBuildSizesInfo(const BlasAABB& blas) const;
    uint32_t shaderGroupHandleSize() const;
    uint32_t shaderGroupHandleAlignment() const;
    uint32_t shaderGroupBaseAlignment() const;
    uint32_t asBufferOffsetAlignment() const;
    uint32_t minAccelerationStructureScratchOffsetAlignment() const;
    VkAccelerationStructureBuildSizesInfoKHR getBuildSizesInfo(const AsBuildInfo& info) const;
    AccelerationStructure createAccelerationStructure(const AsCreateInfo& info) ;
    
};


class Queue {
    VULKAN_CLASS_COMMON2(Queue)
    QueueType _type = queue_max;
public:

    QueueType type() const;

    uint32_t queueFamilyIndex() const;

    uint32_t index() const;

    float priority() const;

    Queue submit(
        CommandBuffer cmdBuffer
    );

    Queue submit(
        std::vector<CommandBuffer> cmdBuffers
    );

    Queue submit(
        std::vector<SubmissionBatchInfo>&& batches
    );

    Queue submit(
        std::vector<SubmissionBatchInfo>&& batches,
        std::optional<Fence> fence
    );

    Queue waitIdle();
};


class CommandPool {
    VULKAN_CLASS_COMMON2(CommandPool)
public:

    QueueType type() const;

    std::vector<CommandBuffer> newCommandBuffers(
        uint32_t count
    );

    CommandBuffer newCommandBuffer();
};


class CommandBuffer {
    VULKAN_CLASS_COMMON2(CommandBuffer)
public:

    QueueType type() const;

    uint32_t queueFamilyIndex() const;

    CommandBuffer submit(uint32_t index=0) const;

    Queue lastSubmittedQueue() const;
    
    void wait() const {
        lastSubmittedQueue().waitIdle();
    }

    CommandBuffer begin(
        VkCommandBufferUsageFlags flags=0
    );

    CommandBuffer end();

    CommandBuffer bindPipeline(
        Pipeline pipeline
    );

    CommandBuffer bindDescSets(
        PipelineLayout layout, 
        VkPipelineBindPoint bindPoint,
        std::vector<DescriptorSet> descSets, 
        uint32_t firstSet=0
    );

    CommandBuffer bindDescSets(
        std::vector<DescriptorSet> descSets, 
        uint32_t firstSet=0
    );

    CommandBuffer setPushConstants(
        PipelineLayout layout, 
        VkShaderStageFlags stageFlags, 
        uint32_t offset, 
        uint32_t size,
        const void* values
    );

    CommandBuffer setPushConstants(
        uint32_t offset, 
        uint32_t size, 
        const void* data
    );

    CommandBuffer barrier(
        std::vector<BarrierInfo> barrierInfos
    );

    CommandBuffer barrier(
        BarrierInfo barrierInfo
    );

	CommandBuffer copyBuffer(
        Buffer src, 
        Buffer dst, 
        uint64_t srcOffset = 0, 
        uint64_t dstOffset = 0, 
        uint64_t size = VK_WHOLE_SIZE
    );

    CommandBuffer copyBuffer(
        BufferRange src,
        BufferRange dst 
    );

    CommandBuffer copyImage(
        Image src,
        Image dst,
        std::vector<CopyRegion> regions = {}
    );

    CommandBuffer copyBufferToImage(
        BufferRange src, 
        Image dst, 
        std::vector<CopyRegion> regions = {}
    );
    
    CommandBuffer copyImageToBuffer(
        Image src, 
        BufferRange dst, 
        std::vector<CopyRegion> regions = {}
    );

    CommandBuffer dispatch(
        uint32_t groupCountX, 
        uint32_t groupCountY=1, 
        uint32_t groupCountZ=1
    );

    CommandBuffer dispatch2(
        uint32_t numThreadsInX, 
        uint32_t numThreadsInY=1, 
        uint32_t numThreadsInZ=1
    );

    CommandBuffer traceRays(
        ShaderBindingTable hitGroupSbt,
        uint32_t width, 
        uint32_t height = 1, 
        uint32_t depth = 1
    );

    CommandBuffer traceRays(
        uint32_t width, 
        uint32_t height = 1, 
        uint32_t depth = 1
    );

    CommandBuffer buildAccelerationStructures(
        const AsBuildInfoInstances& info
    );

    CommandBuffer buildAccelerationStructures(
        const AsBuildInfo& info
    );

    CommandBuffer buildAccelerationStructures(
        const std::vector<AsBuildInfo>& infos
    );
};


class Fence {
    VULKAN_CLASS_COMMON2(Fence)
public:

    VkResult wait(bool autoReset = false, uint64_t timeout=uint64_t(-1)) const;
    void reset() const;
    bool isSignaled() const;
};


class Semaphore {
    VULKAN_CLASS_COMMON2(Semaphore)
public:

    SemaphoreStage operator/(PIPELINE_STAGE stage) const;
    
};


class ShaderModule {
    VULKAN_CLASS_COMMON2(ShaderModule)
public:

    VkPipelineShaderStageCreateInfo stage() const;
    bool hasReflect() const;
    void discardReflect() ;
    PipelineLayoutDesc extractPipelineLayoutDesc() const;

    operator uint64_t() const;
};


class GraphicsPipeline {};


class ComputePipeline {
    VULKAN_CLASS_COMMON2(ComputePipeline)
public:

    PipelineLayout layout() const;
    DescriptorSetLayout descSetLayout(uint32_t setId=0) const;
};


class RaytracingPipeline {
    VULKAN_CLASS_COMMON2(RaytracingPipeline)
public:

    PipelineLayout layout() const;
    DescriptorSetLayout descSetLayout(uint32_t setId0=0) const;
    ShaderGroupHandle getHitGroupHandle(uint32_t groupIndex) const;
    void setHitGroupSbt(ShaderBindingTable sbt);
};


class Buffer {
    VULKAN_CLASS_COMMON2(Buffer)
public:
    
    uint8_t* map(
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE
    );
 
    void flush(
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE
    ) const;

    void invalidate(
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE
    ) const;
    
    void unmap();
    uint64_t size() const;
    VkBufferUsageFlags usage() const;
    VkMemoryPropertyFlags memoryProperties() const;

    VkDescriptorBufferInfo descInfo(
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE
    ) const;

    VkDeviceAddress deviceAddress() const;

    BufferRange operator()(
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE
    );

};


class Image {
    VULKAN_CLASS_COMMON2(Image)
public:

    ImageView view() const;
    ImageView view(ImageViewDesc&& desc) const;

    // operator ImageMemoryBarrier() const;
    // ImageMemoryBarrier operator/(IMAGE_LAYOUT newLayout) const;
    ImageMemoryBarrier operator()(IMAGE_LAYOUT oldLayout, IMAGE_LAYOUT newLayout) const;
};


class ImageView {
    VULKAN_CLASS_COMMON2(ImageView)
public:
};


class Sampler{
    VULKAN_CLASS_COMMON2(Sampler)
public:
};


class DescriptorSetLayout {
    VULKAN_CLASS_COMMON2(DescriptorSetLayout)
public:

    const VkDescriptorSetLayoutBinding& bindingInfo(
        uint32_t bindingId, 
        bool exact=true
    ) const;
        
    // const std::map<uint32_t, VkDescriptorSetLayoutBinding>& bindingInfos() const;
};


class PipelineLayout {
    VULKAN_CLASS_COMMON2(PipelineLayout)
public:

    DescriptorSetLayout descSetLayout(uint32_t setId) const;
};


class DescriptorPool {
    VULKAN_CLASS_COMMON2(DescriptorPool)
public:

    std::vector<DescriptorSet> operator()(
        std::vector<DescriptorSetLayout> layouts
    );

    DescriptorSet operator()(DescriptorSetLayout layout);

    std::vector<DescriptorSet> operator()(
        DescriptorSetLayout layout,
        uint32_t count
    );

    template<typename... Layouts>
    auto operator()(Layouts... layouts) requires (std::is_same_v<Layouts, DescriptorSetLayout> && ...)
    {
        auto sets = (*this)(std::vector<DescriptorSetLayout>{ layouts... });

        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(sets[I]...);
        }(std::index_sequence_for<Layouts...>{});
    }
};


class DescriptorSet {
    VULKAN_CLASS_COMMON2(DescriptorSet)
public:

    DescriptorSet write(
        std::vector<Descriptor> descriptors, 
        uint32_t startBindingId=0, 
        uint32_t startArrayOffset=0
    );

    DescriptorSet operator=(
        std::vector<DescriptorSet>&& data
    );
};

inline DescriptorSet DescriptorPool::operator()(DescriptorSetLayout layout) 
{
    return (*this)(std::vector<DescriptorSetLayout>{layout})[0];
}

inline std::vector<DescriptorSet> DescriptorPool::operator()(DescriptorSetLayout layout, uint32_t count) 
{
    return (*this)(std::vector<DescriptorSetLayout>(count, layout));
}


struct WindowCreateInfo {
    const char* title;
    uint32_t width;
    uint32_t height;
    bool hidden = false;

    Device device;
    VkImageUsageFlags swapChainImageUsage;
    uint32_t minSwapChainImages = 0;
    VkFormat swapChainImageFormat = VK_FORMAT_B8G8R8A8_SRGB;
    VkColorSpaceKHR swapChainImageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkPresentModeKHR preferredPresentMode = VK_PRESENT_MODE_FIFO_KHR;
};


class Window {
    VULKAN_CLASS_COMMON2(Window)

public:
    
    const std::vector<Image>& swapChainImages() const;
    uint32_t acquireNextImageIndex(Semaphore imageAvailableSemaphore) const;
    Image acquireNextImage(Semaphore imageAvailableSemaphore) const;
    void present(Queue queue, std::vector<Semaphore> waitSemaphore, uint32_t imageIndex) const;
    void present(Queue queue, std::vector<Semaphore> waitSemaphore, Image image) const;
    void present(Queue queue, std::vector<Semaphore> waitSemaphore) const;

    bool shouldClose() const;
    void pollEvents() const;

    void setMouseButtonCallback(void (*callback)(int button, int action, double xpos, double ypos));
    void setKeyCallback(void (*callback)(int key, int action, int mods));
    void setCursorPosCallback(void (*callback)(double xpos, double ypos));
    void setScrollCallback(void (*callback)(double xoffset, double yoffset));
};


class AccelerationStructure {
    VULKAN_CLASS_COMMON2(AccelerationStructure)
public:
    VkDeviceAddress deviceAddress() const;
};


///////////////////////////////////////////////////////////////////////////////
struct BindingInfo {
    uint32_t binding;
    DESCRIPTOR_TYPE descriptorType;
    uint32_t descriptorCount;
    SHADER_STAGE stageFlags;

    BindingInfo& operator|=(BindingInfo&& other) {
        if (binding == other.binding 
            && descriptorType == other.descriptorType 
            && descriptorCount == other.descriptorCount)
            stageFlags |= other.stageFlags;
        else
            throw;
        return *this;
    }
};


struct DescriptorSetLayoutDesc {
    std::map<uint32_t, BindingInfo> bindings;

    DescriptorSetLayoutDesc() = default;

    DescriptorSetLayoutDesc(const DescriptorSetLayoutDesc&) = default;
    
    DescriptorSetLayoutDesc(DescriptorSetLayoutDesc&& other) = default;

    DescriptorSetLayoutDesc& operator|=(DescriptorSetLayoutDesc&& other) 
    {
        for (auto& [rId, rBinding] : other.bindings) 
        {
            auto it = bindings.find(rId);
            if (it != bindings.end()) 
                it->second |= std::move(rBinding);
            else 
                bindings.emplace_hint(it, rId, std::move(rBinding));

        }
        return *this;
    }
};


struct PushConstantRange {
    SHADER_STAGE stageFlags;
    uint32_t offset;
    uint32_t size;

    PushConstantRange() : offset(0), size(0), stageFlags(SHADER_STAGE::NONE) {};

    PushConstantRange(uint32_t offset, uint32_t size, SHADER_STAGE stageFlags=SHADER_STAGE::NONE)
    : offset(offset), size(size), stageFlags(stageFlags) {}

    PushConstantRange(uint32_t size, SHADER_STAGE stageFlags=SHADER_STAGE::NONE)
    : offset(0), size(size), stageFlags(stageFlags) {}

    // PushConstantRange&& operator|=(SHADER_STAGE stageFlags) &&
    // {
    //     this->stageFlags |= stageFlags;
    //     return std::move(*this);
    // }

    PushConstantRange& operator|=(PushConstantRange&& other)
    {
        offset = std::min(offset, other.offset);
        size = std::max(offset + size, other.offset + other.size) - offset;
        stageFlags |= other.stageFlags;
        return *this;
    }
};




struct PipelineLayoutDesc {
    std::map<uint32_t, DescriptorSetLayoutDesc> setLayouts;
    /*
    * [VUID-VkPipelineLayoutCreateInfo-pPushConstantRanges-00292]
    * : Any two elements of pPushConstantRanges must not include the same stage in stageFlags.
    * 
    * std::vector<PushConstantRange> pushConstants;  
    *  - this cannot guarantee the above rule
    * 
    * std::map<SHADER_STAGE, PushConstantRange> pushConstants;
    *  - Each SHADER_STAGE key must contain only one bit, which is difficult to use.
    * 
    * PushConstantRange pushConstant;
    *  - It contains one push constant range across all shader stages in the pipeline.
    *  - It may be faster than the stage-granular version when calling vkCmdPushConstants.
    */
    std::unique_ptr<PushConstantRange> pushConstant; // only one push constant range is allowed

    PipelineLayoutDesc() = default;

    PipelineLayoutDesc(const PipelineLayoutDesc& other) 
    : setLayouts(other.setLayouts)
    , pushConstant(other.pushConstant ? std::make_unique<PushConstantRange>(*other.pushConstant) : nullptr)
    {}
    
    PipelineLayoutDesc(PipelineLayoutDesc&& other) = default;

    PipelineLayoutDesc& operator|=(PipelineLayoutDesc&& other)
    {
        for (auto& [rId, rSetLayout] : other.setLayouts) 
        {
            auto it = setLayouts.find(rId);
            if (it != setLayouts.end()) 
                it->second |= std::move(rSetLayout);
            else 
                setLayouts.emplace_hint(it, rId, std::move(rSetLayout));
        }

        if (other.pushConstant) 
        {
            if (pushConstant)
                *pushConstant |= std::move(*other.pushConstant);
            else
                pushConstant = std::move(other.pushConstant);
        }

        return *this;
    }

};


struct BufferDescriptor {
    Buffer buffer;
    uint64_t offset;
    uint64_t size;

    BufferDescriptor(Buffer buffer) 
    : buffer(buffer), offset(0), size(VK_WHOLE_SIZE) {}

    BufferDescriptor(BufferRange range);
};


struct ImageDescriptor {
    std::optional<ImageView> imageView;
    std::optional<Sampler> sampler; 
    IMAGE_LAYOUT imageLayout;

    ImageDescriptor(Image image)
    : imageView(image.view())
    , imageLayout(IMAGE_LAYOUT::MAX_ENUM) {}

    ImageDescriptor(ImageView imageView)
    : imageView(imageView)
    , imageLayout(IMAGE_LAYOUT::MAX_ENUM) {}
};


inline ImageDescriptor&& operator/(ImageDescriptor&& image, Sampler sampler) 
{
    // ASSERT_(!image.sampler);
    image.sampler = sampler;
    return std::move(image);
}

inline ImageDescriptor&& operator/(ImageDescriptor&& image, IMAGE_LAYOUT layout) 
{
    // ASSERT_(image.imageLayout == IMAGE_LAYOUT::MAX_ENUM);
    image.imageLayout = layout;
    return std::move(image);
}


inline std::vector<DescriptorSet> operator,(DescriptorSet lhs, DescriptorSet rhs)
{
    return {lhs, rhs};
}



struct CopyRegion {
    uint64_t bufferOffset=0;
    uint32_t bufferRowLength=0;
    uint32_t bufferImageHeight=0;
    uint32_t offsetX=0;
    uint32_t offsetY=0;
    uint32_t offsetZ=0;
    uint32_t baseLayer=0;
    uint32_t width=0; 
    uint32_t height=0; 
    uint32_t depth=0;
    uint32_t layerCount=0;
};



/**/
constexpr inline VkDescriptorPoolSize operator<=(
    VkDescriptorType type, int count)
{
    return {
        .type = type,
        .descriptorCount = (uint32_t)count,
    };
}





using SpvBlob = std::pair<uint32_t*, size_t>; // (data, size in bytes)


struct ShaderModuleCreateInfo {
    SHADER_STAGE stage;
    std::variant<const char*, SpvBlob> src;
    bool withSpirvReflect = true;
};

using ShaderInput = std::variant<const char*, SpvBlob, ShaderModule>;


template <uint32_t ID, typename T>
struct ConstantID {
    T value;
    ConstantID(T v) : value(v) {}
};

// template <uint32_t ID>
// struct ConstantID<ID, bool> {
//     uint32_t value;
//     ConstantID(bool v) : value(v ? 1u : 0u) {}
// };


template<int ID, class T>
inline auto constant_id(T v)
{
    return ConstantID<ID, T>{v};
}

/*
* VUID-VkSpecializationMapEntry-constantID-00776: 
* If the specialization constant is of type boolean, size must be the byte size of VkBool32.
* And in Vulkan, VkBool32 is defined as uint32_t.
*/
template<int ID>
inline auto constant_id(bool v)
{
    return ConstantID<ID, uint32_t>{ v ? 1u : 0u };
}


class SpecializationConstant {
    std::map<uint32_t, std::vector<uint8_t>> orderedConstants; // key: constantID, value: bytes of the constant value

    mutable std::optional<std::vector<VkSpecializationMapEntry>> cachedMapEntries;
    mutable std::optional<std::vector<uint8_t>> cachedData;
    mutable std::optional<VkSpecializationInfo> cachedSpecInfo;
    
    void buildCache() const;
public:

    template<uint32_t ID, typename T>
    void addConstant(ConstantID<ID, T> constant) 
    {
        auto it = orderedConstants.find(ID);
        if (it != orderedConstants.end()) throw;

        std::vector<uint8_t> newConstant(sizeof(constant.value));
        std::memcpy(newConstant.data(), &constant.value, sizeof(constant.value));
        // *((T*) newConstant.data()) = constant.value; 
        orderedConstants.emplace_hint(it, ID, std::move(newConstant));

        cachedMapEntries.reset();
        cachedData.reset();
        cachedSpecInfo.reset();
    }

    template<typename... ConstantIDs>
    SpecializationConstant(ConstantIDs... constantIDs) 
    {
        (addConstant(constantIDs), ...);
    }
    
    SpecializationConstant() = default;

    SpecializationConstant(const SpecializationConstant& other) 
    : orderedConstants(other.orderedConstants) {}

    SpecializationConstant(SpecializationConstant&& other) 
    : orderedConstants(std::move(other.orderedConstants)) {}
    
    bool empty() const { return orderedConstants.empty(); }

    const VkSpecializationInfo* getInfo() const
    {
        if (empty())  return nullptr;
        buildCache();
        return &cachedSpecInfo.value();
    }

    bool operator==(const SpecializationConstant& other) const noexcept 
    {
        return orderedConstants == other.orderedConstants;
    }

    uint64_t hash() const noexcept;
};


struct ShaderStage {
    std::optional<ShaderInput> shader;
    SpecializationConstant specialization;

    ShaderStage() : shader(std::nullopt), specialization() {}
    ShaderStage(ShaderInput shader, SpecializationConstant&& spec ={})
    : shader(shader), specialization(std::move(spec)) {}

    template<typename ShaderType>
    ShaderStage(ShaderType shader, SpecializationConstant&& spec ={})
    : shader(shader), specialization(std::move(spec)) {}

    template<int ID, typename T>
    ShaderStage operator+(ConstantID<ID, T> constant) && 
    {
        specialization.addConstant(constant);
        return std::move(*this);
    }

    bool operator==(const ShaderStage& other) const noexcept;
};


template<int ID, typename T>
inline ShaderStage operator+(ShaderInput shader, ConstantID<ID, T> constant) 
{
    return ShaderStage(shader) + constant;
}


struct ComputePipelineCreateInfo {
    ShaderStage csStage;
    std::optional<PipelineLayout> layout;
};


struct HitGroup {
    ShaderStage chitStage;
    ShaderStage ahitStage;
    ShaderStage isecStage;
};


struct RaytracingPipelineCreateInfo {
    ShaderStage rgenStage;
    std::vector<ShaderStage> missStages;
    std::vector<HitGroup> hitGroups;
    uint32_t maxRecursionDepth = 1;
    std::optional<PipelineLayout> layout;
    // uint32_t rgenCustomDataSize;
    // uint32_t missCustomDataSize;
};


struct BufferCreateInfo {
    uint64_t size;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags reqMemProps;
};


struct ImageCreateInfo {
    VkImageCreateFlags flags = 0;
    VkFormat format;
    struct Extent {
        uint32_t width;
        uint32_t height = 1;
        uint32_t depth = 1;
    } extent;
    uint32_t arrayLayers = 1;
    VkImageUsageFlags usage;
    bool preInitialized = false; // if true, initialLayout is VK_IMAGE_LAYOUT_PREINITIALIZED, else VK_IMAGE_LAYOUT_UNDEFINED
    VkMemoryPropertyFlags reqMemProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
};


struct ImageViewDesc {
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_MAX_ENUM;
    VkFormat format = VK_FORMAT_MAX_ENUM;
    VkComponentMapping components = {};
    // VkImageSubresourceRange subresourceRange;

    bool operator==(const ImageViewDesc& other) const {
        return viewType == other.viewType &&
               format   == other.format   &&
               components.r == other.components.r &&
               components.g == other.components.g &&
               components.b == other.components.b &&
               components.a == other.components.a;
    }
};


struct SamplerCreateInfo {
    VkFilter magFilter = VK_FILTER_LINEAR;
    VkFilter minFilter = VK_FILTER_LINEAR;
    VkSamplerMipmapMode mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VkSamplerAddressMode addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    float mipLodBias = 0.0f;
    VkBool32 anisotropyEnable = VK_FALSE;
    float maxAnisotropy = 1.0f;
    VkBool32 compareEnable = VK_FALSE;
    VkCompareOp compareOp = VK_COMPARE_OP_ALWAYS;
    float minLod = 0.0f;
    float maxLod = VK_LOD_CLAMP_NONE;
    VkBorderColor borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    VkBool32 unnormalizedCoordinates = VK_FALSE;
};


inline ImageView Image::view() const
{
    return view(ImageViewDesc{});
}


struct DescriptorPoolCreateInfo {
    std::vector<VkDescriptorPoolSize> maxTypes;
    uint32_t maxSets;
};



struct QueueSelector {
    const Device device;
    const uint32_t index;

    QueueSelector(Device device, uint32_t index) 
    : device(device), index(index) {} 

    Queue operator()(CommandBuffer cmdBuffer) const
    {
        return device.queue(cmdBuffer.type(), index);
    }

    Queue submit(CommandBuffer cmdBuffer) const
    {
        return (*this)(cmdBuffer).submit(cmdBuffer);
    }

    Queue submit(std::vector<CommandBuffer> cmdBuffers) const
    {
        return (*this)(cmdBuffers[0]).submit(std::move(cmdBuffers));
    }

    Queue submit(std::vector<SubmissionBatchInfo>&& batches, std::optional<Fence> fence = std::nullopt) const
    {
        return (*this)(std::get<1>(batches[0])[0]).submit(std::move(batches), fence);
    }
};




/*
버퍼 range class가 꼭 필요한가?
버퍼 range 필요 시점:
- vkMapMemory
- vkFlushMappedMemoryRanges, vkInvalidateMappedMemoryRanges
- vkCmdCopyBuffer, vkCmdUpdateBuffer, vkCmdFillBuffer 
- VkDescriptorBufferInfo (vkUpdateDescriptorSets의 인자)
- VkBufferMemoryBarrier 
- VkBufferViewCreateInfo 
*/
struct BufferRange {
    Buffer buffer;
    const uint64_t offset;
    const uint64_t size;

    BufferRange() 
    : buffer({})
    , offset(0)
    , size(0) {};
    
    BufferRange(Buffer buffer, 
        uint64_t offset=0, 
        uint64_t size=VK_WHOLE_SIZE)
    : buffer(buffer)
    , offset(offset)
    , size(size==VK_WHOLE_SIZE ? buffer.size() - offset : size) {}

    BufferRange(const BufferRange&) = default;
    BufferRange(BufferRange&& other) = default;
    
    BufferRange& operator=(const BufferRange& other)
    {
        new (this) BufferRange(other);
        return *this;
    }

    BufferRange& operator=(BufferRange&& other)
    {
        new (this) BufferRange(std::move(other));
        return *this;
    }

    operator bool() const
    {
        return size != 0;
    }

    void flush() const
    {
        buffer.flush(offset, size);
    }

    void invalidate() const
    {
        buffer.invalidate(offset, size);
    }

    VkBufferUsageFlags usage() const
    {
        return buffer.usage();
    }

    VkMemoryPropertyFlags memoryProperties() const
    {
        return buffer.memoryProperties();
    }

    VkDescriptorBufferInfo descInfo() const
    {
        return buffer.descInfo(offset, size);
    }

    VkDeviceAddress deviceAddress() const
    {
        return buffer.deviceAddress() + offset;
    }
};


inline BufferRange Buffer::operator()(uint64_t offset, uint64_t size)
{    
    if (size == VK_WHOLE_SIZE) 
    {
        // ASSERT_(offset < this->size());
        size = this->size() - offset;
    }
    // else ASSERT_(offset + size <= this->size());

    return {*this, offset, size};
}


inline BufferDescriptor::BufferDescriptor(BufferRange range)
: buffer(range.buffer), offset(range.offset), size(range.size) {}



/*
The old layout must either be VK_IMAGE_LAYOUT_UNDEFINED, or match the
current layout of the image subresource range. If the old layout matches the current layout of the
image subresource range, the transition preserves the contents of that range. If the old layout is
VK_IMAGE_LAYOUT_UNDEFINED, the contents of that range may be discarded.
*/
/*
When transitioning the image to VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR or
VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, there is no need to delay subsequent processing,
or perform any visibility operations (as vkQueuePresentKHR performs automatic
visibility operations). To achieve this, the dstAccessMask member of the
VkImageMemoryBarrier should be 0, and the dstStageMask parameter should be
VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
*/
/*
• VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is equivalent to VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT with
  VkAccessFlags2 set to 0 when specified in the second synchronization scope, but equivalent to
  VK_PIPELINE_STAGE_2_NONE in the first scope.
• VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT is equivalent to VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
  with VkAccessFlags2 set to 0 when specified in the first synchronization scope, but equivalent to
  VK_PIPELINE_STAGE_2_NONE in the second scope.
*/
/*
Accesses to the acceleration structure scratch buffers as identified by the
VkAccelerationStructureBuildGeometryInfoKHR::scratchData buffer device addresses must be
synchronized with the VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR pipeline stage and
an access type of (VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR). Accesses to each
VkAccelerationStructureBuildGeometryInfoKHR::srcAccelerationStructure and
VkAccelerationStructureBuildGeometryInfoKHR::dstAccelerationStructure must be synchronized
with the VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR pipeline stage and an access type
of VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR or
VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, as appropriate.
Accesses to other input buffers as identified by any used values of
VkAccelerationStructureGeometryMotionTrianglesDataNV::vertexData,
VkAccelerationStructureGeometryTrianglesDataKHR::vertexData,
VkAccelerationStructureGeometryTrianglesDataKHR::indexData,
VkAccelerationStructureGeometryTrianglesDataKHR::transformData,
VkAccelerationStructureGeometryAabbsDataKHR::data, and
VkAccelerationStructureGeometryInstancesDataKHR::data must be synchronized with the
VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR pipeline stage and an access type of
VK_ACCESS_SHADER_READ_BIT.
*/
struct SYNC_SCOPE {
    struct T {
        PIPELINE_STAGE stage;
        ACCESS access;
        
        bool operator<(const T& other) const {
            if (stage != other.stage) return stage < other.stage;
            return access < other.access;
        }

        // bool operator==(const T& other) const {
        //     return stage == other.stage && access == other.access;
        // }
    } scope;

    SYNC_SCOPE(T scope) : scope(scope) {}
    SYNC_SCOPE(PIPELINE_STAGE stage) : scope({stage, ACCESS::NONE}) {}

    inline static T NONE                = {PIPELINE_STAGE::NONE, ACCESS::NONE};
    inline static T ALL                 = {PIPELINE_STAGE::ALL_COMMANDS, ACCESS::MEMORY_READ | ACCESS::MEMORY_WRITE};
    inline static T ALL_READ            = {PIPELINE_STAGE::ALL_COMMANDS, ACCESS::MEMORY_READ};
    inline static T ALL_WRITE           = {PIPELINE_STAGE::ALL_COMMANDS, ACCESS::MEMORY_WRITE};
    inline static T COMPUTE_READ        = {PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ};
    inline static T COMPUTE_WRITE       = {PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE};
    inline static T RAYTRACING_READ     = {PIPELINE_STAGE::RAY_TRACING_SHADER, ACCESS::SHADER_READ};
    inline static T RAYTRACING_READ_AS  = {PIPELINE_STAGE::RAY_TRACING_SHADER, ACCESS::ACCELERATION_STRUCTURE_READ};
    inline static T RAYTRACING_WRITE    = {PIPELINE_STAGE::RAY_TRACING_SHADER, ACCESS::SHADER_WRITE};
    inline static T TRANSFER_SRC        = {PIPELINE_STAGE::TRANSFER, ACCESS::TRANSFER_READ};
    inline static T TRANSFER_DST        = {PIPELINE_STAGE::TRANSFER, ACCESS::TRANSFER_WRITE};
    inline static T ASBUILD_READ        = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::SHADER_READ};
    inline static T ASBUILD_READ_AS     = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::ACCELERATION_STRUCTURE_READ};
    inline static T ASBUILD_WRITE_AS    = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::ACCELERATION_STRUCTURE_WRITE};
    inline static T ASBUILD_READ_WRITE_AS  = {PIPELINE_STAGE::ACCELERATION_STRUCTURE_BUILD, ACCESS::ACCELERATION_STRUCTURE_READ | ACCESS::ACCELERATION_STRUCTURE_WRITE};
    inline static T PRESENT_SRC         = {(PIPELINE_STAGE)(uint64_t)-1, (ACCESS)(uint64_t)-1};
};

inline SYNC_SCOPE operator,(PIPELINE_STAGE stage, ACCESS access)
{
    return SYNC_SCOPE::T{stage, access};
}


struct MemoryBarrier {
    SYNC_SCOPE srcMask = SYNC_SCOPE::NONE;
    SYNC_SCOPE dstMask = SYNC_SCOPE::NONE;
};

inline MemoryBarrier operator/(SYNC_SCOPE mask1, SYNC_SCOPE mask2)
{
    return {mask1, mask2};
}


struct BufferMemoryBarrier {
    SYNC_SCOPE srcMask = SYNC_SCOPE::NONE;
    SYNC_SCOPE dstMask = SYNC_SCOPE::NONE;
    OwnershipTransferOpType opType = OwnershipTransferOpType::none;
    QueueType pairedQueue = queue_max;
    // const Buffer& buffer;
    // uint64_t offset = 0;
    // uint64_t size = VK_WHOLE_SIZE;
    BufferRange buffer;

    BufferMemoryBarrier(Buffer buffer) : buffer(buffer) {}
    BufferMemoryBarrier(BufferRange buffer) : buffer(buffer) {}
};

inline BufferMemoryBarrier&& operator/(SYNC_SCOPE mask, BufferMemoryBarrier&& barrier)
{
    barrier.srcMask = mask;
    return std::move(barrier);
}

inline BufferMemoryBarrier&& operator/(BufferMemoryBarrier&& barrier, SYNC_SCOPE mask)
{
    barrier.dstMask = mask;
    return std::move(barrier);
}

// inline BufferMemoryBarrier&& operator-(QueueType queueType, BufferMemoryBarrier&& barrier)
// {
//     barrier.opType = OwnershipTransferOpType::acquire;
//     barrier.pairedQueue = queueType;
//     return std::move(barrier);
// }

// inline BufferMemoryBarrier&& operator-(BufferMemoryBarrier&& barrier, QueueType queueType)
// {
//     barrier.opType = OwnershipTransferOpType::release;
//     barrier.pairedQueue = queueType;
//     return std::move(barrier);
// }


struct ImageMemoryBarrier {
    SYNC_SCOPE srcMask = SYNC_SCOPE::NONE;
    SYNC_SCOPE dstMask = SYNC_SCOPE::NONE;
    IMAGE_LAYOUT oldLayout = IMAGE_LAYOUT::UNDEFINED;
    IMAGE_LAYOUT newLayout = IMAGE_LAYOUT::UNDEFINED;
    OwnershipTransferOpType opType = OwnershipTransferOpType::none;
    QueueType pairedQueue = queue_max;
    Image image;
    // VkImageSubresourceRange subresourceRange = {};

    ImageMemoryBarrier(Image image) : image(image) {}
};


// inline Image::operator ImageMemoryBarrier() const 
// { 
//     return {
//         .image = *this,
//     };
// }

// inline ImageMemoryBarrier Image::operator/(IMAGE_LAYOUT newLayout) const 
// { 
//     return {
//         .newLayout = newLayout,
//         .image = *this,
//     };
// }

inline ImageMemoryBarrier Image::operator()(IMAGE_LAYOUT oldLayout, IMAGE_LAYOUT newLayout) const
{ 
    ImageMemoryBarrier barrier(*this);
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    return barrier;
}

inline ImageMemoryBarrier&& operator/(SYNC_SCOPE mask, ImageMemoryBarrier&& barrier)
{
    barrier.srcMask = mask;
    return std::move(barrier);
}

inline ImageMemoryBarrier&& operator/(ImageMemoryBarrier&& barrier, SYNC_SCOPE mask)
{
    barrier.dstMask = mask;
    return std::move(barrier);
}



struct SemaphoreStage {
    const Semaphore sem;
    const PIPELINE_STAGE stage;

    SemaphoreStage(
        Semaphore sem, 
        PIPELINE_STAGE stage=PIPELINE_STAGE::ALL_COMMANDS) 
    : sem(sem), stage(stage) {}
};

inline SemaphoreStage Semaphore::operator/(PIPELINE_STAGE stage) const
{
    return {*this, stage};
}


inline std::vector<SemaphoreStage> operator,(SemaphoreStage sem1, SemaphoreStage sem2)
{
    return {sem1, sem2};
}   

inline std::vector<SemaphoreStage>&& operator,(std::vector<SemaphoreStage>&& sems, SemaphoreStage sem)
{
    sems.push_back(sem);
    return std::move(sems);
}

inline std::vector<CommandBuffer> operator,(CommandBuffer cmdBuffer1, CommandBuffer cmdBuffer2)
{
    return {cmdBuffer1, cmdBuffer2};
}

inline std::vector<CommandBuffer>&& operator,(std::vector<CommandBuffer>&& cmdBuffers, CommandBuffer cmdBuffer)
{
    cmdBuffers.push_back(cmdBuffer);
    return std::move(cmdBuffers);
}

inline SubmissionBatchInfo operator/(SemaphoreStage sem, CommandBuffer cmdBuffer)
{
    return {{sem}, {cmdBuffer}, {}};
}

inline SubmissionBatchInfo operator/(std::vector<SemaphoreStage>&& sems, CommandBuffer cmdBuffer)
{
    return {std::move(sems), {cmdBuffer}, {}};
}

inline SubmissionBatchInfo operator/(CommandBuffer cmdBuffer, SemaphoreStage sem)
{
    return {{}, {cmdBuffer}, {sem}};
}

inline SubmissionBatchInfo operator/(CommandBuffer cmdBuffer, std::vector<SemaphoreStage>&& sems)
{
    return {{}, {cmdBuffer}, std::move(sems)};
}

inline SubmissionBatchInfo operator/(SemaphoreStage sem, std::vector<CommandBuffer>&& cmdBuffers)
{
    return {{sem}, std::move(cmdBuffers), {}};
}

inline SubmissionBatchInfo operator/(std::vector<SemaphoreStage>&& sems, std::vector<CommandBuffer>&& cmdBuffers)
{
    return {std::move(sems), std::move(cmdBuffers), {}};
}

inline SubmissionBatchInfo operator/(std::vector<CommandBuffer>&& cmdBuffers, SemaphoreStage sem)
{
    return {{}, std::move(cmdBuffers), {sem}};
}

inline SubmissionBatchInfo operator/(std::vector<CommandBuffer>&& cmdBuffers, std::vector<SemaphoreStage>&& sems)
{
    return {{}, std::move(cmdBuffers), std::move(sems)};
}

inline SubmissionBatchInfo&& operator/(SubmissionBatchInfo&& batch, SemaphoreStage sem)
{
    std::get<2>(batch).push_back(sem);
    return std::move(batch);
}

inline SubmissionBatchInfo&& operator/(SubmissionBatchInfo&& batch, std::vector<SemaphoreStage>&& sems)
{
    std::get<2>(batch) = std::move(sems);
    return std::move(batch);
}

struct Waiting {};
inline void waiting(Waiting w){}

struct Submitting {
private:
    friend Submitting operator<<(Queue queue, SubmissionBatchInfo&& batch);
    friend Submitting operator<<(Queue queue, CommandBuffer cmdBuffer);
    friend Submitting operator<<(Queue queue, std::vector<CommandBuffer>&& cmdBuffers);
    friend Submitting&& operator<<(Submitting&& submitting, SubmissionBatchInfo&& batch);
    friend Submitting&& operator<<(Submitting&& submitting, CommandBuffer cmdBuffer);
    friend Submitting&& operator<<(Submitting&& submitting, std::vector<CommandBuffer>&& cmdBuffers);
    friend void operator<<(Submitting&& submitting, Fence fence);
    friend void operator<<(Submitting&& submitting, void(Waiting));

    Submitting() = delete;
    Submitting(const Submitting&) = delete;
    Submitting(Submitting&&) = delete;

    Submitting(Queue queue, SubmissionBatchInfo&& batch) : queue(queue)
    {
        batches.emplace_back(std::move(batch));
    }
    Queue queue;
    std::vector<SubmissionBatchInfo> batches;
    bool isWaiting = false;
    std::optional<Fence> fence;
    
public:
    ~Submitting() { 
        queue.submit(std::move(batches), fence); 
       
        if (isWaiting) {
            queue.waitIdle();
        }
    }
};


inline Submitting operator<<(Queue queue, CommandBuffer cmdBuffer)
{
    return Submitting(queue, {{}, {cmdBuffer}, {}});
}

inline Submitting operator<<(Queue queue, std::vector<CommandBuffer>&& cmdBuffers)
{
    return Submitting(queue, {{}, std::move(cmdBuffers), {}});
}

inline Submitting operator<<(Queue queue, SubmissionBatchInfo&& batch)
{
    return Submitting(queue, std::move(batch));
}

inline Submitting operator<<(QueueSelector queueSelector, CommandBuffer cmdBuffer)
{
    return operator<<(queueSelector(cmdBuffer), cmdBuffer);
}

inline Submitting operator<<(QueueSelector queueSelector, std::vector<CommandBuffer>&& cmdBuffers)
{
    return operator<<(queueSelector(cmdBuffers[0]), std::move(cmdBuffers));
}

inline Submitting operator<<(QueueSelector queueSelector, SubmissionBatchInfo&& batch)
{
    return operator<<(queueSelector(std::get<1>(batch)[0]), std::move(batch));
}

inline Submitting&& operator<<(Submitting&& submitting, CommandBuffer cmdBuffer)
{
    submitting.batches.emplace_back(
        std::vector<SemaphoreStage>{}, 
        std::vector<CommandBuffer>{cmdBuffer}, 
        std::vector<SemaphoreStage>{});
    return std::move(submitting);
}

inline Submitting&& operator<<(Submitting&& submitting, std::vector<CommandBuffer>&& cmdBuffers)
{
    submitting.batches.emplace_back(
        std::vector<SemaphoreStage>{}, 
        std::move(cmdBuffers), 
        std::vector<SemaphoreStage>{});
    return std::move(submitting);
}

inline Submitting&& operator<<(Submitting&& submitting, SubmissionBatchInfo&& batch)
{   
    submitting.batches.emplace_back(std::move(batch));
    return std::move(submitting);
}

inline void operator<<(Submitting&& submitting, Fence fence)
{
    submitting.fence = fence;
}

inline void operator<<(Submitting&& submitting, void(Waiting))
{
    submitting.isWaiting = true;
}



struct ShaderBindingTable {
    Buffer buffer;
    uint32_t recordSize;
    uint32_t numRecords;
};



struct AABB 
{
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
};


struct StridedBuffer {
    BufferRange buffer;
    uint32_t stride;
};


struct AsCreateInfo {
    VkAccelerationStructureTypeKHR asType;  
    BufferRange internalBuffer;     // offset and size required for AS data storage
    uint64_t size;                  // Although it could be fed via internalBuffer.size, it is explicitly specified to prevent user mistakes
};


struct AsBuildInfoTriangles {
    VkBuildAccelerationStructureFlagsKHR buildFlags;
    AccelerationStructure srcAs;
    AccelerationStructure dstAs;
    BufferRange scratchBuffer;      // offset required, size ignored

    struct Geometry {
        VkGeometryFlagsKHR flags;
        uint32_t triangleCount;
        uint32_t vertexCount;

        StridedBuffer vertexInput;
        StridedBuffer indexInput;  // stride must be 0, 2, or 4
        BufferRange transformBuffer;
    };
    std::vector<Geometry> geometries;

    struct {
        StridedBuffer vertexInput;
        StridedBuffer indexInput;
        // StridedBuffer transformInput;
    } common;
};


struct AsBuildInfoAabbs {
    VkBuildAccelerationStructureFlagsKHR buildFlags;
    AccelerationStructure srcAs;
    AccelerationStructure dstAs;
    BufferRange scratchBuffer;    

    struct Geometry {
        VkGeometryFlagsKHR flags;
        uint32_t aabbCount;
        StridedBuffer aabbInput;
    };
    std::vector<Geometry> geometries;

    struct {
        StridedBuffer aabbInput;
    } common;
};


struct AsBuildInfoInstances {
    VkBuildAccelerationStructureFlagsKHR buildFlags;
    AccelerationStructure srcAs;
    AccelerationStructure dstAs;
    BufferRange scratchBuffer;    

    // uint32_t instanceCount;
    // StridedBuffer instanceInput;

    struct Geometry {
        VkGeometryFlagsKHR flags;
        uint32_t instanceCount;
        StridedBuffer instanceInput;
    };

    std::vector<Geometry> geometries;
    struct {
        StridedBuffer instanceInput;
    } common;
};


struct AsBuildInfo {
    VkBuildAccelerationStructureFlagsKHR buildFlags;
    VkGeometryTypeKHR geometryType;
    std::vector<uint32_t> primitiveCounts; // primitive count for each geometry
    AccelerationStructure srcAs;
    AccelerationStructure dstAs;
    BufferRange scratchBuffer;    

    struct Triangles {
        StridedBuffer vertexInput;
        StridedBuffer indexInput;
        std::vector<uint32_t> vertexCounts;

        struct Geometry {
            VkGeometryFlagsKHR flags;
            StridedBuffer vertexInput;
            StridedBuffer indexInput;  // stride must be 0, 2, or 4
            BufferRange transformBuffer;
        };
        std::vector<Geometry> eachGeometry;
    };

    struct Aabbs{
        StridedBuffer aabbInput;

        struct Geometry {
            VkGeometryFlagsKHR flags;
            StridedBuffer aabbInput;
        };
        std::vector<Geometry> eachGeometry;
    };

    struct Instances{
        BufferRange instanceInput;
    };

    using Inputs = std::variant<Triangles, Aabbs, Instances>;
    Inputs inputs;
};


} // namespace ve



namespace std {
    template<>
    struct hash<ve::ShaderStage> {
        size_t operator()(const ve::ShaderStage& stage) const noexcept;
    };
} 

inline auto alignTo = [](auto value, auto alignment) -> decltype(value) {
    return (value + (decltype(value))alignment - 1) & ~((decltype(value))alignment - 1);
};


#endif // VULKAN_APP_H