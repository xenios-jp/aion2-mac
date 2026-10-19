#import "gpu.h"
#include <CommonCrypto/CommonDigest.h>
#include <unistd.h>
#include <math.h>

@implementation SrGpu {
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *_pso;
    id<MTLLibrary> _lib;
    BOOL _dirty;
    id<MTLBinaryArchive> _archive;
    NSURL *_archiveURL;
    BOOL _archiveDirty, _archiveFresh;
    id<MTLBuffer> _tanh;
}
- (instancetype)initWithDevice:(id<MTLDevice>)dev libraryData:(NSData *)data error:(NSString **)err {
    if (!(self = [super init])) return nil;
    _device = dev;
    NSError *e = nil;
    dispatch_data_t dd = dispatch_data_create(data.bytes, data.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    _lib = [dev newLibraryWithData:dd error:&e];
    if (!_lib) { if (err) *err = [NSString stringWithFormat:@"shader library: %@", e.localizedDescription]; return nil; }
    _queue = [dev newMTL4CommandQueue];
    _allocator = [dev newCommandAllocator];
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    _residency = [dev newResidencySetWithDescriptor:rd error:&e];
    if (!_queue || !_allocator || !_residency) { if (err) *err = @"cannot create Metal 4 queue or residency set"; return nil; }
    _pso = [NSMutableDictionary new];
    return self;
}
- (void)enableArchiveForLibrary:(NSData *)lib fallbackDirectory:(NSString *)fallbackDir {
    /* one archive file per (OS build, GPU name, shader library hash) in the user's cache directory; any failure just disables caching */
    unsigned char dig[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(lib.bytes, (CC_LONG)lib.length, dig);
    NSMutableString *key = [NSMutableString string];
    for (int i = 0; i < 8; i++) [key appendFormat:@"%02x", dig[i]];
    NSString *ident = [NSString stringWithFormat:@"%@|%@", NSProcessInfo.processInfo.operatingSystemVersionString, _device.name];
    CC_SHA256(ident.UTF8String, (CC_LONG)strlen(ident.UTF8String), dig);
    for (int i = 0; i < 6; i++) [key appendFormat:@"%02x", dig[i]];
    NSString *dir = nil;
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *cands[2] = {[[NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) firstObject] stringByAppendingPathComponent:@"metalsr"],
                          fallbackDir ? [fallbackDir stringByAppendingPathComponent:@".metalsr-cache"] : nil};
    for (int i = 0; i < 2 && !dir; i++) {
        if (!cands[i]) continue;
        [fm createDirectoryAtPath:cands[i] withIntermediateDirectories:YES attributes:nil error:nil];
        if ([fm isWritableFileAtPath:cands[i]]) dir = cands[i];
    }
    if (!dir) return;
    _archiveURL = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:[key stringByAppendingString:@".archive"]]];
    MTLBinaryArchiveDescriptor *d = [MTLBinaryArchiveDescriptor new];
    if ([[NSFileManager defaultManager] fileExistsAtPath:_archiveURL.path]) d.url = _archiveURL;
    _archiveFresh = d.url == nil;
    NSError *e = nil;
    _archive = [_device newBinaryArchiveWithDescriptor:d error:&e];
    if (!_archive && d.url) { d.url = nil; _archive = [_device newBinaryArchiveWithDescriptor:d error:&e]; }   /* unreadable: start over */
}
- (id<MTLBuffer>)tanhTable {
    if (!_tanh) {
        _tanh = [self buffer:65536 * 4];
        float *t = _tanh.contents;
        for (uint32_t b = 0; b < 65536; b++) {
            uint32_t e = (b >> 10) & 31, m = b & 1023;
            double v;
            if (e == 31) v = m ? 0.0 : ((b & 0x8000) ? -INFINITY : INFINITY);     /* NaN codes never reach the head (relu'd inputs) */
            else {
                double mag = e == 0 ? ldexp((double)m, -24) : ldexp(1.0 + m / 1024.0, (int)e - 15);
                v = (b & 0x8000) ? -mag : mag;
            }
            t[b] = (float)tanh(v);
        }
    }
    return _tanh;
}
- (void)useLibrary:(id<MTLLibrary>)lib {
    _lib = lib;
    [_pso removeAllObjects];
    _archive = nil;
}
- (void)saveArchive {
    if (!_archive || !_archiveDirty || !_archiveURL) return;
    NSURL *tmp = [NSURL fileURLWithPath:[[_archiveURL.path stringByDeletingPathExtension] stringByAppendingFormat:@".tmp%d.archive", getpid()]];
    NSError *e = nil;
    if ([_archive serializeToURL:tmp error:&e]) {
        int rr = rename(tmp.fileSystemRepresentation, _archiveURL.fileSystemRepresentation);
        (void)rr;
        _archiveDirty = NO;
    } else { [[NSFileManager defaultManager] removeItemAtURL:tmp error:nil]; }
}
- (id<MTLComputePipelineState>)pipeline:(NSString *)name {
    id<MTLComputePipelineState> p = _pso[name];
    if (p) return p;
    id<MTLFunction> f = [_lib newFunctionWithName:name];
    if (!f) { NSLog(@"missing kernel %@", name); return nil; }
    NSError *e = nil;
    if (_archive) {
        MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
        d.computeFunction = f;
        d.binaryArchives = @[_archive];
        p = _archiveFresh ? nil : [_device newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:&e];
        if (!p) {   /* miss: compile, remember in the archive */
            if ([_archive addComputePipelineFunctionsWithDescriptor:d error:&e]) _archiveDirty = YES;

            p = [_device newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&e];
        }
    } else p = [_device newComputePipelineStateWithFunction:f error:&e];
    if (!p) { NSLog(@"pipeline %@: %@", name, e); return nil; }
    _pso[name] = p;
    return p;
}
- (void)track:(id<MTLAllocation>)a { [_residency addAllocation:a]; _dirty = YES; }
- (id<MTLBuffer>)buffer:(size_t)bytes {
    id<MTLBuffer> b = [_device newBufferWithLength:bytes ? bytes : 16 options:MTLResourceStorageModeShared];
    memset(b.contents, 0, b.length);
    [self track:b];
    return b;
}
- (id<MTLTexture>)texture:(MTLPixelFormat)fmt w:(int)w h:(int)h {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [_device newTextureWithDescriptor:d];
    [self track:t];
    return t;
}
- (id<MTLTexture>)privateTexture:(MTLPixelFormat)fmt w:(int)w h:(int)h {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [_device newTextureWithDescriptor:d];
    [self track:t];
    return t;
}
- (void)commitResidency { if (_dirty) { [_residency commit]; _dirty = NO; } }
@end

@implementation SrEnc {
    SrGpu *_gpu;
    id<MTL4ArgumentTable> _table;
    id<MTLBuffer> _ring;
    size_t _ringPos;
}
- (instancetype)initWithGpu:(SrGpu *)gpu encoder:(id<MTL4ComputeCommandEncoder>)enc ring:(id<MTLBuffer>)ring {
    if (!(self = [super init])) return nil;
    _gpu = gpu;
    MTL4ArgumentTableDescriptor *d = [MTL4ArgumentTableDescriptor new];
    d.maxBufferBindCount = 16; d.maxTextureBindCount = 16;
    NSError *e = nil;
    _table = [gpu.device newArgumentTableWithDescriptor:d error:&e];
    if (enc) [self attachEncoder:enc ring:ring];
    return self;
}
- (void)attachEncoder:(id<MTL4ComputeCommandEncoder>)enc ring:(id<MTLBuffer>)ring {
    _enc = enc; _ring = ring; _ringPos = 0; _dispatches = 0; _barriers = 0;
    [enc setArgumentTable:_table];
}
- (void)resetRing { _ringPos = 0; }
- (void)dispatch:(id<MTLComputePipelineState>)pso args:(const SrArg *)args count:(int)n
         threadgroups:(MTLSize)tgs threads:(MTLSize)tpt {
    int nb = 0, nt = 0;
    for (int i = 0; i < n; i++) {
        const SrArg *a = &args[i];
        if (a->kind == 0) [_table setAddress:((id<MTLBuffer>)a->obj).gpuAddress + a->offset atIndex:nb++];
        else if (a->kind == 1) [_table setTexture:((id<MTLTexture>)a->obj).gpuResourceID atIndex:nt++];
        else {
            size_t at = (_ringPos + 255) & ~(size_t)255;
            NSAssert(at + a->len <= _ring.length, @"constants ring exhausted");
            memcpy((char *)_ring.contents + at, a->bytes, a->len);
            [_table setAddress:_ring.gpuAddress + at atIndex:nb++];
            _ringPos = at + a->len;
        }
    }
    [_enc setComputePipelineState:pso];
    [_enc dispatchThreadgroups:tgs threadsPerThreadgroup:tpt];
    _dispatches++;
}
- (void)dispatchGrid:(MTLSize)g pso:(id<MTLComputePipelineState>)pso args:(const SrArg *)args count:(int)n threads:(MTLSize)t {
    MTLSize tgs = MTLSizeMake((g.width + t.width - 1) / t.width, (g.height + t.height - 1) / t.height, (g.depth + t.depth - 1) / t.depth);
    [self dispatch:pso args:args count:n threadgroups:tgs threads:t];
}
- (void)barrier {
    [_enc barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
    _barriers++;
}
@end

NSString *sr_run_sync(SrGpu *gpu, void (^block)(SrEnc *e)) {
    static id<MTLBuffer> ring;
    if (!ring) ring = [gpu buffer:1 << 20];
    [gpu commitResidency];
    id<MTL4CommandBuffer> cb = [gpu.device newCommandBuffer];
    [gpu.allocator reset];
    [cb beginCommandBufferWithAllocator:gpu.allocator];
    [cb useResidencySet:gpu.residency];
    id<MTL4ComputeCommandEncoder> en = [cb computeCommandEncoder];
    SrEnc *e = [[SrEnc alloc] initWithGpu:gpu encoder:en ring:ring];
    block(e);
    [en endEncoding];
    [cb endCommandBuffer];
    id<MTLSharedEvent> ev = [gpu.device newSharedEvent];
    id<MTL4CommandBuffer> list[1] = {cb};
    [gpu.queue commit:list count:1];
    [gpu.queue signalEvent:ev value:1];
    if (![ev waitUntilSignaledValue:1 timeoutMS:60000]) return @"GPU timeout";
    return nil;
}
