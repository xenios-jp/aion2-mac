/* Per-call orchestration (spec fg.orchestration): schedule, clears (FH2), rect copies (F28), history rotation (FH1). */
#import <Foundation/Foundation.h>
#include <string.h>
#include "passes_impl.h"
#include "metalfgconst.h"

NSArray<NSString *> *metalfg_pass_kernel_names(void) {
    return @[];
}

void metalfg_fill_tex(MetalFGEngineImpl *e, MetalFGEnc *enc, id<MTLTexture> t, float r, float g, float b, float a) {
    MetalFGFillParams p = {{r, g, b, a}, 0, 0};
    [enc bindTexture:t at:0];
    [enc setBytes:&p length:sizeof p at:0];
    metalfg_go2d(enc, e.pso[@"fill_tex"], t.width, t.height);
}

void metalfg_fill_tex_u32(MetalFGEngineImpl *e, MetalFGEnc *enc, id<MTLTexture> t, uint32_t v) {
    MetalFGFillParams p = {{0, 0, 0, 0}, 0, v};
    [enc bindTexture:t at:0];
    [enc setBytes:&p length:sizeof p at:0];
    metalfg_go2d(enc, e.pso[@"fill_tex_u32"], t.width, t.height);
}

void metalfg_fill_buf_u32(MetalFGEngineImpl *e, MetalFGEnc *enc, id<MTLBuffer> buf, uint32_t count32, uint32_t v) {
    MetalFGFillParams p = {{0, 0, 0, 0}, count32, v};
    [enc bindBuffer:buf offset:0 at:1];
    [enc setBytes:&p length:sizeof p at:0];
    metalfg_go1d(enc, e.pso[@"fill_buf_u32"], count32);
}

typedef struct { uint32_t src[2], dst[2], size[2]; } RectParams;

void metalfg_rect_copy(MetalFGEngineImpl *e, MetalFGEnc *enc, id<MTLTexture> src, id<MTLTexture> dst, uint32_t sx, uint32_t sy,
                   uint32_t dx, uint32_t dy, uint32_t w, uint32_t h) {
    if (!w || !h) return;
    RectParams p = {{sx, sy}, {dx, dy}, {w, h}};
    [enc bindTexture:src at:0];
    [enc bindTexture:dst at:1];
    [enc setBytes:&p length:sizeof p at:0];
    metalfg_go2d(enc, e.pso[@"rect_copy"], w, h);
}

@implementation MetalFGEngineImpl (Run)

/* F28: the display rectangle of colour to the output, plus the strips around the rectangle (colour texel at the same
   coordinates where the colour texture has one). With strips_only the display rectangle is left out: in a generating call the synthesis (F22, F23) writes
   every texel of the display rectangle of out_interp after the prefill, so the prefill there cannot be seen. */
- (void)f28:(MetalFGEnc *)enc frame:(const MetalFGFrameInfo *)f to:(void *)outp stripsOnly:(BOOL)strips_only {
    id<MTLTexture> col = (__bridge id<MTLTexture>)f->colour, out = (__bridge id<MTLTexture>)outp;
    uint32_t W = grids.W, H = grids.H;
    uint32_t w = MIN(f->content_w, W), h = MIN(f->content_h, H);
    uint32_t ox = f->out_off[0], oy = f->out_off[1];
    if (!strips_only) metalfg_rect_copy(self, enc, col, out, f->colour_off[0], f->colour_off[1], ox, oy, w, h);
    uint32_t ow = (uint32_t)out.width, oh = (uint32_t)out.height;
    if (oy > 0) metalfg_rect_copy(self, enc, col, out, 0, 0, 0, 0, ow, oy);
    if (oy + H < oh) metalfg_rect_copy(self, enc, col, out, 0, oy + H, 0, oy + H, ow, oh - (oy + H));
    if (ox > 0) metalfg_rect_copy(self, enc, col, out, 0, oy, 0, oy, ox, H);
    if (ox + W < ow) metalfg_rect_copy(self, enc, col, out, ox + W, oy, ox + W, oy, ow - (ox + W), H);
}

/* FH2: start values of fg.resources on a reset call. */
- (void)fh2:(MetalFGEnc *)enc {
    id<MTLTexture> t;
#if !defined(METALFG_CTL_KEEP_STATE)                       /* control: the first network's recurrent state is not cleared on a reset call */
    if ((t = [self tex:@"st.feat" old:NO])) metalfg_fill_tex(self, enc, t, 0.5f, 0, 0, 0);
#endif
    if ((t = [self tex:@"dyn.mot" old:YES])) metalfg_fill_tex(self, enc, t, 1, 0, 0, 0);
    if ((t = [self tex:@"dep.mot" old:YES])) metalfg_fill_tex_u32(self, enc, t, 0);
    if ((t = [self tex:@"col.full" old:YES])) metalfg_fill_tex(self, enc, t, 0, 0, 0, 1);
    if ((t = [self tex:@"ui.full" old:YES])) metalfg_fill_tex(self, enc, t, 0, 0, 0, 1);
    if ((t = [self tex:@"ui.full" old:NO])) metalfg_fill_tex(self, enc, t, 0, 0, 0, 1);
    for (NSString *n in @[ @"uim.feat", @"col.feat", @"err.feat", @"dis.feat" ])
        if ((t = [self tex:n old:YES])) metalfg_fill_tex(self, enc, t, 0, 0, 0, 0);
    if ((t = [self tex:@"dep2.mot" old:YES])) metalfg_fill_tex_u32(self, enc, t, 0);
    if ((t = [self tex:@"dep2.mot" old:NO])) metalfg_fill_tex_u32(self, enc, t, 0);
}

- (void)fillScatter:(NSString *)name {
}

- (MetalFGDepthConst)depthConst:(const MetalFGFrameInfo *)f {
    MetalFGDepthConst d = {info.depth_scale, info.depth_partition, info.depth_min_sep, 1e-7f, (f->flags & METALFG_FRAME_DEPTH_REVERSED) ? 1u : 0u};
    return d;
}

- (void)encodeF04:(MetalFGEnc *)enc frame:(const MetalFGFrameInfo *)f plan:(const MetalFGCallPlan *)p {
    id<MTLTexture> col = (__bridge id<MTLTexture>)f->colour;
    id<MTLTexture> hud = f->hudless ? (__bridge id<MTLTexture>)f->hudless : col;
    id<MTLTexture> cov = f->ui_alpha ? (__bridge id<MTLTexture>)f->ui_alpha : (f->ui_colour ? (__bridge id<MTLTexture>)f->ui_colour : col);
    BOOL ui = (info.flags & METALFG_CREATE_UI_RECOMPOSE) != 0;
    MetalFGF04Args a = {(uint32_t)grids.W, (uint32_t)grids.H, {f->colour_off[0], f->colour_off[1]}, (uint32_t)p->f04, ui ? 1u : 0u,
                    f->ui_alpha ? 0u : 1u, (uint32_t)grids.df, params.th_ui, params.th_uia};
    [self.chain f04:enc args:a colour:col hudless:hud coverage:cov colOld:[self tex:@"col.full" old:YES] colCur:[self tex:@"col.full" old:NO]
              uiCur:(ui ? [self tex:@"ui.full" old:NO] : nil) colFeat:[self tex:@"col.feat" old:NO] uim:[self tex:@"uim.feat" old:NO]
                 dY:[self tex:@"dY.feat" old:NO]];
}

- (void)encodeP2C:(MetalFGEnc *)enc frame:(const MetalFGFrameInfo *)f plan:(const MetalFGCallPlan *)p {
    MetalFGP2CArgs a;
    memset(&a, 0, sizeof a);
    a.Wm = (uint32_t)grids.Wm; a.Hm = (uint32_t)grids.Hm; a.depthW = (float)METALFG_DEPTH_W(f); a.depthH = (float)METALFG_DEPTH_H(f);
    a.H = (float)grids.H; a.d_dyn = (float)params.d_dyn; a.d_sim = (float)params.d_sim; a.th_cam = params.th_cam;
    a.D = [self depthConst:f];
    memcpy(a.c2p, camMats, sizeof a.c2p); memcpy(a.p2c, camMats + 16, sizeof a.p2c);
    a.writeNsc = p->dist ? 0 : 1;
    [self.chain p2c:enc args:a sc:[self buf:@"sc.pv" old:NO] hb:[self buf:@"hb.mot" old:NO] depCur:[self tex:@"dep.mot" old:NO]
             depOld:[self tex:@"dep.mot" old:YES] dynCur:[self tex:@"dyn.mot" old:NO] dynOld:[self tex:@"dyn.mot" old:YES]
                mvo:[self tex:@"mvo.mot" old:NO] mvf:[self tex:@"mvf.mot" old:NO] nsc:(p->dist ? nil : [self tex:@"nsc.mot" old:NO])];
}

/* F18 phase ph (0: the current-call projection into sc.ic, 1: the previous-frame projection into sc.ip), rows y0 .. y1 - 1 */
- (void)encodeF18:(MetalFGEnc *)enc frame:(const MetalFGFrameInfo *)f plan:(const MetalFGCallPlan *)p phase:(int)ph rows:(uint32_t)y0 to:(uint32_t)y1 {
    BOOL D = p->dist;
    MetalFGF18Args a;
    memset(&a, 0, sizeof a);
    a.Wm = (uint32_t)grids.Wm; a.Hm = (uint32_t)grids.Hm; a.H = (float)grids.H; a.d_sim = (float)params.d_sim;
    a.th_zn = params.th_zn; a.Z_sep = info.depth_min_sep; a.sp_min = params.sp_min; a.sp_max = params.sp_max; a.D = [self depthConst:f];
    float t = p->t, omt = 1.0f - t;
    float sc[2] = {omt * (float)grids.Wm, omt * (float)grids.Hm}, sp[2] = {t, t};
    if (ph == 0)
        [self.chain f18:enc args:a field:[self tex:D ? @"mvod.mot" : @"mvo.mot" old:NO] hscale:sc dep:[self tex:D ? @"dep2.mot" : @"dep.mot" old:NO]
                     sc:[self buf:@"sc.ic" old:NO] useZn:YES rows:y0 to:y1];
    else
        [self.chain f18:enc args:a field:[self tex:D ? @"mvfd.mot" : @"mvf.mot" old:NO] hscale:sp dep:[self tex:D ? @"dep2.mot" : @"dep.mot" old:YES]
                     sc:[self buf:@"sc.ip" old:NO] useZn:NO rows:y0 to:y1];
}

- (void)encodeF18:(MetalFGEnc *)enc frame:(const MetalFGFrameInfo *)f plan:(const MetalFGCallPlan *)p {
    BOOL D = p->dist;
    MetalFGF18Args a;
    memset(&a, 0, sizeof a);
    a.Wm = (uint32_t)grids.Wm; a.Hm = (uint32_t)grids.Hm; a.H = (float)grids.H; a.d_sim = (float)params.d_sim;
    a.th_zn = params.th_zn; a.Z_sep = info.depth_min_sep; a.sp_min = params.sp_min; a.sp_max = params.sp_max; a.D = [self depthConst:f];
    float t = p->t, omt = 1.0f - t;
    float sc[2] = {omt * (float)grids.Wm, omt * (float)grids.Hm}, sp[2] = {t, t};
    [self.chain f18:enc args:a field:[self tex:D ? @"mvod.mot" : @"mvo.mot" old:NO] hscale:sc dep:[self tex:D ? @"dep2.mot" : @"dep.mot" old:NO]
                 sc:[self buf:@"sc.ic" old:NO] useZn:YES];
    [enc skipNextBarrier];   /* the previous-frame projection writes other cells and reads other images than the one before */
    [self.chain f18:enc args:a field:[self tex:D ? @"mvfd.mot" : @"mvf.mot" old:NO] hscale:sp dep:[self tex:D ? @"dep2.mot" : @"dep.mot" old:YES]
                 sc:[self buf:@"sc.ip" old:NO] useZn:NO];
}

- (void)runCall:(MetalFGEnc *)enc frame:(const MetalFGFrameInfo *)f plan:(const MetalFGCallPlan *)p {
    id<MTLFence> fence = (__bridge id<MTLFence>)f->fence;
    [enc waitFence:fence];

    /* First call: give every image and buffer a defined content (zero), so that seams that write nothing yet
       cannot feed uninitialised memory to the passes that exist. The FH2 start values follow in the schedule. */
    if (callCounter == 0) {
        for (MetalFGSlot *s in self.slots.allValues) {
            if (strcmp(s.desc.name, "menu.state") == 0) continue;
            for (int side = 0; side < (s.desc.pair ? 2 : 1); side++) {
                id<MTLTexture> t = side ? s.tex1 : s.tex0;
                id<MTLBuffer> b = side ? s.buf1 : s.buf0;
                if (t && t.pixelFormat == MTLPixelFormatR32Uint) metalfg_fill_tex_u32(self, enc, t, 0);
                else if (t) metalfg_fill_tex(self, enc, t, 0, 0, 0, 0);
                else if (b) metalfg_fill_buf_u32(self, enc, b, (uint32_t)(b.length / 4), 0);
            }
        }
    }

    /* bdf.mot.old is cleared when this call carries a field and the previous call did not */
    if (p->dist && !prevHadField) {
        id<MTLTexture> t = [self tex:@"bdf.mot" old:YES];
        if (t) metalfg_fill_tex(self, enc, t, 0, 0, 0, 0);
    }
    /* a call without a field while bit 2 is set zero-fills the current reduced field */
    if (p->dist_call_without_field) {
        id<MTLTexture> t = [self tex:@"bdf.mot" old:NO];
        if (t) metalfg_fill_tex(self, enc, t, 0, 0, 0, 0);
    }

    /* camera matrices for the passes that need them (the previous-to-current scatter and the intermediate scatter) */
    metalfg_camera_matrices(f->world_to_view, f->view_to_clip, prevV, prevP, f->fov_deg, f->aspect, f->near_plane, f->far_plane,
                        (f->flags & METALFG_FRAME_DEPTH_REVERSED) ? 1u : 0u, p->reset ? 1u : 0u, camMats);

    const char *tags[48];
    int n = metalfg_schedule(p, info.flags, &grids, tags);
    int f13_count = 0;
    /* fg.orchestration section 5: F18 depends on none of F15..F17. When the first network runs in this call, F18's scatter is cut into row slices that
       follow F15 and each dispatch of F16 and F17 without a barrier of their own (the atomic minimum makes the result independent of the order) */
    BOOL weave = NO;
    for (int i = 0; i < n; i++) if (!strcmp(tags[i], "F16")) weave = YES;
#if defined(METALFG_CTL_KEY_RECOMPUTE)
    weave = NO;                                        /* control: the payload pass needs the keys of every row, so the two passes run whole */
#endif
    const uint32_t Hm = (uint32_t)grids.Hm, ns = 8;        /* slices per phase (4 to 24 measured; 8 to 16 best) */
    __block uint32_t sliceNext = 0;
    void (^slice)(void) = ^{
        if (sliceNext >= 2 * ns) return;
        const uint32_t ph = sliceNext / ns, i = sliceNext % ns;
        sliceNext++;
        [enc skipNextBarrier];
        [self encodeF18:enc frame:f plan:p phase:(int)ph rows:Hm * i / ns to:Hm * (i + 1) / ns];
    };
    for (int i = 0; i < n; i++) {
        const char *t = tags[i];
        /* fg.orchestration section 5: {F04, F05, F06, F02/F03} are mutually independent, and so is F07's prefill of them; one barrier joins them before
           the F07 scatter. (F01 of a reset call needs its detector, so it keeps its barrier; FH2 is never directly before them: F01 sits between.) */
        const char *prev = i ? tags[i - 1] : "";
        if ((!strcmp(t, "F04") || !strcmp(t, "F05") || !strcmp(t, "F06")) &&
            (!strcmp(prev, "F01") || !strcmp(prev, "F02") || !strcmp(prev, "F03") || !strcmp(prev, "F04") || !strcmp(prev, "F05") || !strcmp(prev, "F06")))
            [enc skipNextBarrier];
        if (!strcmp(t, "F07") && (!strcmp(prev, "F04") || !strcmp(prev, "F05") || !strcmp(prev, "F06"))) [enc skipNextBarrier];
        if (!strcmp(t, "FH2")) {
            [self fh2:enc];
        } else if (!strcmp(t, "F07")) {
            /* scatter buffer prefill: both 32-bit halves read 0x7FFFFFFF */
            id<MTLBuffer> b = [self buf:@"sc.pv" old:NO];
            metalfg_fill_buf_u32(self, enc, b, (uint32_t)(grids.Wm * grids.Hm * 2), 0x7FFFFFFFu);
        } else if (!strcmp(t, "F08")) {
            [self encodeP2C:enc frame:f plan:p];
        } else if (!strcmp(t, "F04")) {
            [self encodeF04:enc frame:f plan:p];
        } else if (!strcmp(t, "F14")) {
            [self.chain f14:enc colFeat:[self tex:@"col.feat" old:NO] err:[self tex:@"err.feat" old:NO]];
        } else if (!strcmp(t, "F13")) {
            BOOL D = p->dist;
            if (f13_count == 0) {
                metalfg_fill_tex(self, enc, [self tex:@"dis.feat" old:NO], 1, 0, 0, 0);
                if (D) metalfg_fill_tex(self, enc, [self tex:@"nsc.mot" old:NO], 1, 0, 0, 0);
                [self.chain f13:enc field:[self tex:D ? @"mvfd.mot" : @"mvf.mot" old:NO] out:[self tex:@"dis.feat" old:NO] normalised:NO
                            Wm:(uint32_t)grids.Wm Hm:(uint32_t)grids.Hm];
            } else {
                [self.chain f13:enc field:[self tex:@"mvod.mot" old:NO] out:[self tex:@"nsc.mot" old:NO] normalised:YES
                            Wm:(uint32_t)grids.Wm Hm:(uint32_t)grids.Hm];
            }
            f13_count++;
        } else if (!strcmp(t, "F18")) {
            if (weave) {                                       /* the slices that did not find a dispatch to follow */
                enc.filler = nil;
                for (uint32_t ph = 0; ph < 2; ph++) {
                    const uint32_t i0 = sliceNext > ph * ns ? MIN(sliceNext - ph * ns, ns) : 0;
                    if (i0 >= ns) continue;
                    [enc skipNextBarrier];
                    [self encodeF18:enc frame:f plan:p phase:(int)ph rows:Hm * i0 / ns to:Hm];
                }
                sliceNext = 2 * ns;
            } else {
                metalfg_fill_buf_u32(self, enc, [self buf:@"sc.ic" old:NO], (uint32_t)(grids.Wm * grids.Hm * 2), 0xFFFFFFFFu);
                metalfg_fill_buf_u32(self, enc, [self buf:@"sc.ip" old:NO], (uint32_t)(grids.Wm * grids.Hm * 2), 0xFFFFFFFFu);
                [self encodeF18:enc frame:f plan:p];
            }
        } else if (!strcmp(t, "F19")) {
            MetalFGF18Args a;
            memset(&a, 0, sizeof a);
            a.Wm = (uint32_t)grids.Wm; a.Hm = (uint32_t)grids.Hm; a.Wf = (uint32_t)grids.Wf; a.Hf = (uint32_t)grids.Hf;
#if defined(METALFG_CTL_FEAT_WEIGHTS)
            [self.chain f19feat:enc args:a Wm:(uint32_t)grids.Wm Hm:(uint32_t)grids.Hm scP:[self buf:@"sc.ip" old:NO] scC:[self buf:@"sc.ic" old:NO]
                            lgP:[self tex:@"lgP.feat" old:NO] lgC:[self tex:@"lgC.feat" old:NO] bw:self.ctlBwFeat];
#else
            [self.chain f19:enc args:a scP:[self buf:@"sc.ip" old:NO] scC:[self buf:@"sc.ic" old:NO] lgP:[self tex:@"lgP.feat" old:NO]
                        lgC:[self tex:@"lgC.feat" old:NO] bw:[self tex:@"bw.mot" old:NO]];
#endif
        } else if (!strcmp(t, "F20")) {
            BOOL ui = (info.flags & METALFG_CREATE_UI_RECOMPOSE) != 0;
            MetalFGF20Args a;
            memset(&a, 0, sizeof a);
            a.scIp = [self buf:@"sc.ip" old:NO]; a.scIc = [self buf:@"sc.ic" old:NO];
            a.n2a = [self buf:@"n2a" old:NO]; a.n2b = [self buf:@"n2b" old:NO]; a.n2h = [self buf:@"n2h" old:NO];
            a.colOld = [self tex:@"col.full" old:YES]; a.colCur = [self tex:@"col.full" old:NO];
            a.bw = [self tex:@"bw.mot" old:NO]; a.err = [self tex:@"err.feat" old:NO];
#if defined(METALFG_CTL_FEAT_WEIGHTS)
            a.bw = self.ctlBwFeat;
#endif
            a.uiOld = ui ? [self tex:@"ui.full" old:YES] : nil; a.uiCur = ui ? [self tex:@"ui.full" old:NO] : nil;
            a.candA = [self tex:@"candA.full" old:NO]; a.candB = [self tex:@"candB.full" old:NO]; a.hole = [self tex:@"hole.full" old:NO];
            a.W = (uint32_t)grids.W; a.H = (uint32_t)grids.H; a.Wm = (uint32_t)grids.Wm; a.Hm = (uint32_t)grids.Hm;
            a.Nw = (uint32_t)grids.Nw; a.Nh = (uint32_t)grids.Nh; a.s = (uint32_t)grids.s; a.ui = ui ? 1 : 0;
            [self.blend blendPack:enc args:a];
        } else if (!strcmp(t, "F21")) {
            [self.net2 encode:enc a:[self buf:@"n2a" old:NO] b:[self buf:@"n2b" old:NO] h:[self buf:@"n2h" old:NO] t:p->t out:[self buf:@"n2out" old:NO]];
        } else if (!strcmp(t, "F22") || !strcmp(t, "F23")) {
            /* F23 stands for the whole chain F23..F26, which MetalFGOutput encodes as one unit; F24 to F26 are then no-ops */
            id<MTLBuffer> n2 = [self buf:@"n2out" old:NO];
            id<MTLTexture> a = [self tex:@"candA.full" old:NO], b = [self tex:@"candB.full" old:NO], h = [self tex:@"hole.full" old:NO];
            id<MTLTexture> out = (__bridge id<MTLTexture>)f->out_interp;
            if (!strcmp(t, "F22"))
                [self.output direct:enc n2out:n2 candA:a candB:b hole:h out:out origin:f->out_off Nw:(uint32_t)grids.Nw Nh:(uint32_t)grids.Nh s:(uint32_t)grids.s];
            else
                [self.output chain:enc n2out:n2 candA:a candB:b hole:h out:out origin:f->out_off Nw:(uint32_t)grids.Nw Nh:(uint32_t)grids.Nh s:(uint32_t)grids.s];
#if defined(METALFG_CTL_BLEND)
            /* control: the per-channel mean of the previous and the current real frame replaces the generated frame */
            {
                RectParams rp = {{0, 0}, {f->out_off[0], f->out_off[1]}, {(uint32_t)grids.W, (uint32_t)grids.H}};
                [enc bindTexture:[self tex:@"col.full" old:YES] at:0];
                [enc bindTexture:[self tex:@"col.full" old:NO] at:1];
                [enc bindTexture:out at:2];
                [enc setBytes:&rp length:sizeof rp at:0];
                metalfg_go2d(enc, self.pso[@"ctl_blend"], grids.W, grids.H);
            }
#endif
        } else if (!strcmp(t, "F05")) {
            MetalFGF05Args a = {(uint32_t)grids.Wm, (uint32_t)grids.Hm, {f->depth_off[0], f->depth_off[1]}, {METALFG_DEPTH_W(f), METALFG_DEPTH_H(f)},
                            {f->motion_off[0], f->motion_off[1]}, {METALFG_MOTION_W(f), METALFG_MOTION_H(f)}, f->motion_scale_x, f->motion_scale_y,
                            (f->flags & METALFG_FRAME_DEPTH_REVERSED) ? 1u : 0u, 1e-7f};
            [self.motion f05:enc depth:(__bridge id<MTLTexture>)f->depth motion:(__bridge id<MTLTexture>)f->motion
                         dep:[self tex:@"dep.mot" old:NO] mvo:[self tex:@"mvo.mot" old:NO] args:a];
        } else if (!strcmp(t, "F06")) {
            uint32_t sz[2] = {f->distortion_size[0] ? f->distortion_size[0] : f->content_w, f->distortion_size[1] ? f->distortion_size[1] : f->content_h};
            [self.motion f06:enc field:(__bridge id<MTLTexture>)f->distortion origin:f->distortion_off size:sz
                         out:[self tex:@"bdf.mot" old:NO] Wm:(uint32_t)grids.Wm Hm:(uint32_t)grids.Hm];
        } else if (!strcmp(t, "F12")) {
            [self.motion f12:enc bdfCur:[self tex:@"bdf.mot" old:NO] bdfOld:[self tex:@"bdf.mot" old:YES] mvo:[self tex:@"mvo.mot" old:NO]
                         mvf:[self tex:@"mvf.mot" old:NO] dep:[self tex:@"dep.mot" old:NO] mvod:[self tex:@"mvod.mot" old:NO]
                         mvfd:[self tex:@"mvfd.mot" old:NO] dep2:[self tex:@"dep2.mot" old:NO] Wm:(uint32_t)grids.Wm Hm:(uint32_t)grids.Hm];
        } else if (!strcmp(t, "F02") || !strcmp(t, "F03")) {
            MetalFGDetectArgs a = {METALFG_GX, METALFG_GY, f->content_w, f->content_h, f->colour_off[0], f->colour_off[1],
                               params.mt, !strcmp(t, "F02") ? params.mfa : params.mfh};
            id<MTLBuffer> flag = f->skip_flag ? (__bridge id<MTLBuffer>)f->skip_flag : nil;
            id<MTLBuffer> mstate = [self buf:@"menu.state" old:NO];
            if (!strcmp(t, "F02")) {
                id<MTLTexture> cov = (__bridge id<MTLTexture>)(f->ui_alpha ? f->ui_alpha : f->ui_colour);
                [self.menu detectAlpha:enc state:mstate flag:flag coverage:cov channel:(f->ui_alpha ? 0 : 3) args:a];
            } else {
                [self.menu detectHudless:enc state:mstate flag:flag colour:(__bridge id<MTLTexture>)f->colour
                                 hudless:(__bridge id<MTLTexture>)f->hudless args:a];
            }
        } else if (!strcmp(t, "F01")) {
            if (f->skip_flag) [self.menu clearFlag:enc flag:(__bridge id<MTLBuffer>)f->skip_flag];
        } else if (!strcmp(t, "F27")) {
#if defined(METALFG_CTL_NO_OVERRIDE)
            continue;                                      /* control: detection runs and writes the flag byte, the frame is never replaced */
#endif
            uint32_t sz[2] = {MIN(f->content_w, (uint32_t)grids.W), MIN(f->content_h, (uint32_t)grids.H)};
            [self.menu override:enc state:[self buf:@"menu.state" old:NO] colour:(__bridge id<MTLTexture>)f->colour
                            out:(__bridge id<MTLTexture>)f->out_interp srcOrigin:f->colour_off dstOrigin:f->out_off size:sz];
        } else if (!strcmp(t, "F15")) {
#if defined(METALFG_CTL_NO_DISOCC)
            metalfg_fill_tex(self, enc, [self tex:@"dis.feat" old:NO], 0, 0, 0, 0);   /* control: both masks are zero before they are used */
            metalfg_fill_tex(self, enc, [self tex:@"nsc.mot" old:NO], 0, 0, 0, 0);
#endif
            if (weave) {                                       /* F18's prefills run beside the last pass before F15; F15's barrier covers them */
                [enc skipNextBarrier];
                metalfg_fill_buf_u32(self, enc, [self buf:@"sc.ic" old:NO], (uint32_t)(grids.Wm * grids.Hm * 2), 0xFFFFFFFFu);
                [enc skipNextBarrier];
                metalfg_fill_buf_u32(self, enc, [self buf:@"sc.ip" old:NO], (uint32_t)(grids.Wm * grids.Hm * 2), 0xFFFFFFFFu);
            }
            BOOL D = p->dist;
            MetalFGF15Args a = {[self tex:@"col.feat" old:YES], [self tex:@"col.feat" old:NO], [self tex:@"dY.feat" old:NO],
                            [self tex:@"err.feat" old:YES], [self tex:@"nsc.mot" old:NO], [self tex:@"dis.feat" old:NO],
                            [self tex:@"dis.feat" old:YES], [self tex:@"st.feat" old:NO],
                            [self tex:D ? @"mvod.mot" : @"mvo.mot" old:NO], [self tex:D ? @"mvfd.mot" : @"mvf.mot" old:NO],
                            [self tex:@"err.feat" old:NO], [self tex:@"ms.feat" old:NO], [self buf:@"n1in" old:NO],
                            (uint32_t)grids.Wf, (uint32_t)grids.Hf, (uint32_t)grids.w1, (uint32_t)grids.h1, (uint32_t)grids.ox,
                            (uint32_t)grids.oy, (uint32_t)grids.Wm, (uint32_t)grids.Hm};
            if (weave) enc.filler = slice;
            [self.io f15:enc args:a];
        } else if (!strcmp(t, "F16")) {
            [self.net1 encode:enc in:[self buf:@"n1in" old:NO] out:[self buf:@"n1out" old:NO]];
        } else if (!strcmp(t, "F17")) {
            MetalFGF17Args a = {[self tex:@"ms.feat" old:NO], [self tex:@"uim.feat" old:YES], [self tex:@"uim.feat" old:NO],
                            [self tex:@"lgP.feat" old:NO], [self tex:@"lgC.feat" old:NO], [self tex:@"st.feat" old:NO],
                            [self buf:@"n1out" old:NO], (uint32_t)grids.Wf, (uint32_t)grids.Hf, (uint32_t)grids.w1,
                            (uint32_t)grids.h1, (uint32_t)grids.ox, (uint32_t)grids.oy, (uint32_t)(p->steering ? 1 : 0),
                            params.b_steer, params.th_mask};
            [self.io f17:enc args:a];
        } else if (!strcmp(t, "F28")) {
            [self f28:enc frame:f to:f->out_interp stripsOnly:!p->reset];
            if (f->out_real) [self f28:enc frame:f to:f->out_real stripsOnly:NO];
        }
        /* every other pass is a seam: see docs/STATUS.md */
        [enc stamp:t];
    }
#if defined(METALFG_CTL_PERF_SPIN)
    /* control 1 of fg.performance_r1: one extra dispatch in the timed path behind in-order barriers; its length is METALFG_CTL_SPIN_ITERS iterations */
    {
        static uint32_t iters = 0;
        if (!iters) { const char *v = getenv("METALFG_CTL_SPIN_ITERS"); iters = v ? (uint32_t)strtoul(v, NULL, 0) : 1000000u; }
        static id<MTLBuffer> sink;
        if (!sink) sink = [self.device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        [enc barrier];
        [enc setPipeline:self.pso[@"ctl_spin"]];
        [enc setBytes:&iters length:4 at:0];
        [enc bindBuffer:sink offset:0 at:1];
        [enc dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerGroup:MTLSizeMake(32, 1, 1)];
        [enc barrier];
        [enc stamp:"ctl_spin"];
    }
#endif
    enc.filler = nil;
    [enc updateFence:fence];
    [enc end];
    lastDispatches = enc.dispatchCount;
    lastBarriers = enc.barrierCount;

    prevHadField = p->dist ? 1 : 0;
    metalfg_call_commit(&state, p, f);
    callCounter++;
    if (p->advance) {
        memcpy(prevV, f->world_to_view, sizeof prevV);
        memcpy(prevP, f->view_to_clip, sizeof prevP);
        [self rotate];
    }
}

/* FH1: the current-frame images take the history role; no data moves. */
- (void)rotate {
    for (MetalFGSlot *s in self.slots.allValues)
        if (s.desc.pair) s.cur = 1 - s.cur;
}

@end
