/* Synthetic named model parameters for tests (the real values only ever come from the extractor's file). */
#ifndef SR_TESTPARAMS_H
#define SR_TESTPARAMS_H
#include "../src/model.h"

static inline SrParams test_params(void) {
    SrParams p;
    p.r_wide = 2.5; p.r_fast = 1.8; p.phi_lo = 1.0 / 4096; p.phi_hi = 1.0 / 2048; p.kappa_mv = 8.0;
    p.sigma_min = 0.3; p.sigma_span = 0.7; p.rho_max = 0.9;
    p.q_a = 1.0; p.q_clip = 1.5; p.q_b = (1.5 * (1.0 + 1.5 * 1.5) - 1.0) / (1.5 * 1.5);   /* q(q_clip) = 1 */
    p.g_beta = 1.0; p.alpha_0 = 0.5; p.key_ae = 0.18; p.eps_luma = 1e-4; p.n_luma_min = 16;
    return p;
}

/* the synthetic parameter values of sr.head_filter_r3 (head-tile-bound): r_wide below 3/2 so that a 3/2 width ratio selects path W */
static inline SrParams test_params_r3(void) {
    SrParams p;
    p.r_wide = 1.25; p.r_fast = 1.5; p.phi_lo = 1.0 / 1024; p.phi_hi = 0.25; p.kappa_mv = 64.0;
    p.sigma_min = 0.25; p.sigma_span = 1.5; p.rho_max = 0.875;
    p.q_a = 4.0; p.q_b = 3.0; p.q_clip = 2.0;
    p.g_beta = 1.0; p.alpha_0 = 1.0 / 16; p.key_ae = 0.25; p.eps_luma = 1.0 / 1048576; p.n_luma_min = 64;
    return p;
}
#endif
