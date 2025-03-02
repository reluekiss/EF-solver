#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#include <pthread.h>

#define ARENA_IMPLEMENTATION
#include "arena.h"

#define NX 64
#define NY 64
#define NZ 64
#define DT 0.001
#define STEPS 1000
#define Lx 1.0
#define Ly 1.0
#define Lz 1.0
#define N (NX*NY*NZ)
#define IDX(i,j,k) (((i)*NY + (j))*NZ + (k))

// Gamma-driver damping parameter and constraint damping constant.
#define ETA 1.0
#define KAPPA 0.1

#define NUM_THREADS 16

typedef enum {
    METRIC_MINKOWSKI,
    METRIC_SCHWARZSCHILD
} MetricType;

typedef struct {
    // 3-metric and extrinsic curvature.
    double *gamma[3][3];
    double *K[3][3];
    // Lapse, shift, and auxiliary field for gamma-driver.
    double *alpha;
    double *beta[3];
    double *B[3];
} Fields;

// Global arenas.
Arena mainArena = {0};
Arena tempArena = {0};

void allocate_fields(Fields *f) {
    int n = N;
    f->alpha = arena_alloc(&mainArena, sizeof(double)*n);
    for (int d = 0; d < 3; d++) {
        f->beta[d] = arena_alloc(&mainArena, sizeof(double)*n);
        f->B[d] = arena_alloc(&mainArena, sizeof(double)*n);
    }
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            f->gamma[a][b] = arena_alloc(&mainArena, sizeof(double)*n);
            f->K[a][b] = arena_alloc(&mainArena, sizeof(double)*n);
        }
    }
}

// Initialize fields for a given metric.
void initialize_fields(Fields *f, MetricType metric) {
    double M = 1.0; // mass parameter for Schwarzschild
    for (int i = 0; i < NX; i++) {
        double x = ((double)i/NX - 0.5)*Lx;
        for (int j = 0; j < NY; j++) {
            double y = ((double)j/NY - 0.5)*Ly;
            for (int k = 0; k < NZ; k++) {
                double z = ((double)k/NZ - 0.5)*Lz;
                int idx = IDX(i,j,k);
                double r = sqrt(x*x + y*y + z*z);
                switch (metric) {
                case METRIC_MINKOWSKI:
                    f->alpha[idx] = 1.0;
                    for (int d = 0; d < 3; d++)
                        f->beta[d][idx] = 0.0;
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++) {
                            f->gamma[a][b][idx] = (a == b) ? 1.0 : 0.0;
                            f->K[a][b][idx] = 0.0;
                        }
                    break;
                case METRIC_SCHWARZSCHILD: {
                    double psi = (r < 1e-6) ? 1.0 : (1.0 + M/(2*r));
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++) {
                            f->gamma[a][b][idx] = (a == b) ? pow(psi, 4) : 0.0;
                            f->K[a][b][idx] = 0.0;
                        }
                    f->alpha[idx] = (r < 1e-6) ? 1.0 : ((1.0 - M/(2*r))/(1.0 + M/(2*r)));
                    for (int d = 0; d < 3; d++)
                        f->beta[d][idx] = 0.0;
                }
                    break;
                default:
                    break;
                }
            }
        }
    }
}

// Spectral derivative (in dimension 'dim': 0=x,1=y,2=z) using FFTW.
void spectral_derivative(double *in, double *out, int dim) {
    fftw_complex *fft_data = fftw_malloc(sizeof(fftw_complex)*N);
    fftw_plan plan_forward  = fftw_plan_dft_r2c_3d(NX, NY, NZ, in, fft_data, FFTW_ESTIMATE);
    fftw_plan plan_backward = fftw_plan_dft_c2r_3d(NX, NY, NZ, fft_data, out, FFTW_ESTIMATE);
    fftw_execute(plan_forward);
    int i, j, k;
    double kx, ky, kz, k_val;
    for (i = 0; i < NX; i++){
        kx = (i < NX/2) ? (2*M_PI*i/Lx) : (2*M_PI*(i-NX)/Lx);
        for (j = 0; j < NY; j++){
            ky = (j < NY/2) ? (2*M_PI*j/Ly) : (2*M_PI*(j-NY)/Ly);
            for (k = 0; k < NZ; k++){
                kz = (k < NZ/2) ? (2*M_PI*k/Lz) : (2*M_PI*(k-NZ)/Lz);
                int idx = IDX(i,j,k);
                if(dim == 0) k_val = kx;
                else if(dim == 1) k_val = ky;
                else k_val = kz;
                double a = fft_data[idx][0], b = fft_data[idx][1];
                fft_data[idx][0] = -b * k_val;
                fft_data[idx][1] =  a * k_val;
            }
        }
    }
    fftw_execute(plan_backward);
    for (int n = 0; n < N; n++)
        out[n] /= N;
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(fft_data);
}

void invert3x3(double m[3][3], double inv[3][3]) {
    double det = m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
               - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
               + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    if (fabs(det) < 1e-12) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                inv[i][j] = (i == j) ? 1.0 : 0.0;
        return;
    }
    double invdet = 1.0 / det;
    inv[0][0] =  (m[1][1]*m[2][2]-m[1][2]*m[2][1]) * invdet;
    inv[0][1] = -(m[0][1]*m[2][2]-m[0][2]*m[2][1]) * invdet;
    inv[0][2] =  (m[0][1]*m[1][2]-m[0][2]*m[1][1]) * invdet;
    inv[1][0] = -(m[1][0]*m[2][2]-m[1][2]*m[2][0]) * invdet;
    inv[1][1] =  (m[0][0]*m[2][2]-m[0][2]*m[2][0]) * invdet;
    inv[1][2] = -(m[0][0]*m[1][2]-m[0][2]*m[1][0]) * invdet;
    inv[2][0] =  (m[1][0]*m[2][1]-m[1][1]*m[2][0]) * invdet;
    inv[2][1] = -(m[0][0]*m[2][1]-m[0][1]*m[2][0]) * invdet;
    inv[2][2] =  (m[0][0]*m[1][1]-m[0][1]*m[1][0]) * invdet;
}

/*
  compute_Ricci computes the spatial Ricci tensor R_ij from the spatial metric γ_{ij}.
  It uses spectral_derivative() to compute derivatives.
  (Quadratic Γ-terms are omitted.)
*/
void compute_Ricci(Fields *f, double *R[3][3]) {
    double *dgamma[3][3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
         for (int d = 0; d < 3; d++) {
             dgamma[i][j][d] = arena_alloc(&tempArena, sizeof(double)*N);
             spectral_derivative(f->gamma[i][j], dgamma[i][j][d], d);
         }
      }
    }
    
    double *Gamma[3][3][3];
    for (int k = 0; k < 3; k++) {
      for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
            Gamma[k][i][j] = arena_alloc(&tempArena, sizeof(double)*N);
         }
      }
    }
    
    for (int idx = 0; idx < N; idx++) {
        double g[3][3], g_inv[3][3];
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
             g[i][j] = f->gamma[i][j][idx];
        invert3x3(g, g_inv);
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                double sum = 0.0;
                for (int l = 0; l < 3; l++) {
                    double term = dgamma[j][l][i][idx] + dgamma[i][l][j][idx] - dgamma[i][j][l][idx];
                    sum += g_inv[k][l] * term;
                }
                Gamma[k][i][j][idx] = 0.5 * sum;
            }
          }
        }
    }
    
    double *dGamma[3][3][3][3];
    for (int d = 0; d < 3; d++) {
      for (int k = 0; k < 3; k++) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            dGamma[d][k][i][j] = arena_alloc(&tempArena, sizeof(double)*N);
            spectral_derivative(Gamma[k][i][j], dGamma[d][k][i][j], d);
          }
        }
      }
    }
    
    for (int idx = 0; idx < N; idx++) {
       for (int i = 0; i < 3; i++) {
         for (int j = 0; j < 3; j++) {
            double term1 = 0.0, term2 = 0.0;
            for (int k = 0; k < 3; k++) {
              term1 += dGamma[k][k][i][j][idx];
              term2 += dGamma[j][k][i][k][idx];
            }
            R[i][j][idx] = term1 - term2;
         }
       }
    }
}

typedef struct {
    int start;
    int end;
    double *d2alpha;       // Precomputed second derivative array (size N)
    double *F_alpha;       // F->alpha (size N)
    double *R_component;   // R_tensor component for this (a,b) (size N)
    double *F_K_component; // F->K[a][b] (size N)
    double traceK;
    double *dF_K_component; // dF->K[a][b] (size N) to update
} DerivThreadData;

void *compute_extrinsic_thread(void *arg) {
    DerivThreadData *data = (DerivThreadData *) arg;
    for (int i = data->start; i < data->end; i++) {
        data->dF_K_component[i] = - data->d2alpha[i] +
            data->F_alpha[i] * ( data->R_component[i] + data->traceK * data->F_K_component[i] );
    }
    return NULL;
}

// Compute derivatives of the fields and store in dF.
void compute_derivatives(const Fields *F, Fields *dF) {
    int i, a, b;
    double traceK;
    /* Lapse evolution: dF->alpha = -2 * F->alpha * trace(K) */
    for (i = 0; i < N; i++){
        traceK = F->K[0][0][i] + F->K[1][1][i] + F->K[2][2][i];
        dF->alpha[i] = -2 * F->alpha[i] * traceK;
    }
    /* Metric evolution: dF->gamma = -2 * F->alpha * F->K */
    for (a = 0; a < 3; a++){
        for (b = 0; b < 3; b++){
            for (i = 0; i < N; i++){
                dF->gamma[a][b][i] = -2 * F->alpha[i] * F->K[a][b][i];
            }
        }
    }
    /* Compute Ricci tensor into temporary arrays (R_tensor[3][3]) */
    double *R_tensor[3][3];
    for (a = 0; a < 3; a++){
        for (b = 0; b < 3; b++){
            R_tensor[a][b] = arena_alloc(&tempArena, sizeof(double)*N);
        }
    }
    compute_Ricci((Fields *)F, R_tensor);
    
    /* Parallelize extrinsic curvature evolution for each (a,b).
       We first compute d2alpha (i.e. second derivative D_aD_b alpha) serially for each (a,b),
       then split the loop over N among NUM_THREADS.
    */
    for (a = 0; a < 3; a++){
        for (b = 0; b < 3; b++){
            double *temp = arena_alloc(&tempArena, sizeof(double)*N);
            double *d2alpha_arr = arena_alloc(&tempArena, sizeof(double)*N);
            spectral_derivative(F->alpha, temp, a);
            spectral_derivative(temp, d2alpha_arr, b);
            // Create thread data and launch threads for indices [0, N)
            pthread_t threads[NUM_THREADS];
            DerivThreadData threadData[NUM_THREADS];
            int chunk = N / NUM_THREADS;
            for (int t = 0; t < NUM_THREADS; t++) {
                threadData[t].start = t * chunk;
                threadData[t].end = (t == NUM_THREADS - 1) ? N : (t + 1) * chunk;
                threadData[t].d2alpha = d2alpha_arr;
                threadData[t].F_alpha = F->alpha;
                threadData[t].R_component = R_tensor[a][b];
                threadData[t].F_K_component = F->K[a][b];
                threadData[t].traceK = traceK;
                threadData[t].dF_K_component = dF->K[a][b];
                pthread_create(&threads[t], NULL, compute_extrinsic_thread, &threadData[t]);
            }
            for (int t = 0; t < NUM_THREADS; t++) {
                pthread_join(threads[t], NULL);
            }
        }
    }
    /* Gamma-driver shift conditions (serial) */
    for (a = 0; a < 3; a++){
        for (i = 0; i < N; i++){
            dF->beta[a][i] = 0.75 * F->B[a][i];
            dF->B[a][i] = -ETA * F->B[a][i];
        }
    }
}

typedef struct {
    int start;
    int end;
    Fields *F;
    Fields *k1;
    Fields *k2;
    Fields *k3;
    Fields *k4;
} CombineData;

void* combine_thread(void *arg) {
    CombineData *data = (CombineData*) arg;
    for (int i = data->start; i < data->end; i++) {
        for (int a = 0; a < 3; a++){
            for (int b = 0; b < 3; b++){
                data->F->gamma[a][b][i] += (DT/6.0) * ( data->k1->gamma[a][b][i]
                    + 2 * data->k2->gamma[a][b][i]
                    + 2 * data->k3->gamma[a][b][i]
                    + data->k4->gamma[a][b][i] );
                data->F->K[a][b][i] += (DT/6.0) * ( data->k1->K[a][b][i]
                    + 2 * data->k2->K[a][b][i]
                    + 2 * data->k3->K[a][b][i]
                    + data->k4->K[a][b][i] );
            }
        }
        data->F->alpha[i] += (DT/6.0) * ( data->k1->alpha[i]
                    + 2 * data->k2->alpha[i]
                    + 2 * data->k3->alpha[i]
                    + data->k4->alpha[i] );
        for (int a = 0; a < 3; a++){
            data->F->beta[a][i] += (DT/6.0) * ( data->k1->beta[a][i]
                    + 2 * data->k2->beta[a][i]
                    + 2 * data->k3->beta[a][i]
                    + data->k4->beta[a][i] );
            data->F->B[a][i] += (DT/6.0) * ( data->k1->B[a][i]
                    + 2 * data->k2->B[a][i]
                    + 2 * data->k3->B[a][i]
                    + data->k4->B[a][i] );
        }
    }
    return NULL;
}

// RK4 integration step using global tempArena for temporary allocations.
void RK4_step(Fields *F) {
    Fields k1, k2, k3, k4, F_temp;
    allocate_fields(&k1);
    allocate_fields(&k2);
    allocate_fields(&k3);
    allocate_fields(&k4);
    allocate_fields(&F_temp);
    int a, b, i;
    // k1 = f(F)
    compute_derivatives(F, &k1);
    // F_temp = F + 0.5*DT*k1
    for (a = 0; a < 3; a++){
        for (b = 0; b < 3; b++){
            for (i = 0; i < N; i++){
                F_temp.gamma[a][b][i] = F->gamma[a][b][i] + 0.5 * DT * k1.gamma[a][b][i];
                F_temp.K[a][b][i]     = F->K[a][b][i]     + 0.5 * DT * k1.K[a][b][i];
            }
        }
    }
    for (i = 0; i < N; i++){
        F_temp.alpha[i] = F->alpha[i] + 0.5 * DT * k1.alpha[i];
        for (a = 0; a < 3; a++){
            F_temp.beta[a][i] = F->beta[a][i] + 0.5 * DT * k1.beta[a][i];
            F_temp.B[a][i]    = F->B[a][i]    + 0.5 * DT * k1.B[a][i];
        }
    }
    compute_derivatives(&F_temp, &k2);
    // F_temp = F + 0.5*DT*k2
    for (a = 0; a < 3; a++){
        for (b = 0; b < 3; b++){
            for (i = 0; i < N; i++){
                F_temp.gamma[a][b][i] = F->gamma[a][b][i] + 0.5 * DT * k2.gamma[a][b][i];
                F_temp.K[a][b][i]     = F->K[a][b][i]     + 0.5 * DT * k2.K[a][b][i];
            }
        }
    }
    for (i = 0; i < N; i++){
        F_temp.alpha[i] = F->alpha[i] + 0.5 * DT * k2.alpha[i];
        for (a = 0; a < 3; a++){
            F_temp.beta[a][i] = F->beta[a][i] + 0.5 * DT * k2.beta[a][i];
            F_temp.B[a][i]    = F->B[a][i]    + 0.5 * DT * k2.B[a][i];
        }
    }
    compute_derivatives(&F_temp, &k3);
    // F_temp = F + DT*k3
    for (a = 0; a < 3; a++){
        for (b = 0; b < 3; b++){
            for (i = 0; i < N; i++){
                F_temp.gamma[a][b][i] = F->gamma[a][b][i] + DT * k3.gamma[a][b][i];
                F_temp.K[a][b][i]     = F->K[a][b][i]     + DT * k3.K[a][b][i];
            }
        }
    }
    for (i = 0; i < N; i++){
        F_temp.alpha[i] = F->alpha[i] + DT * k3.alpha[i];
        for (a = 0; a < 3; a++){
            F_temp.beta[a][i] = F->beta[a][i] + DT * k3.beta[a][i];
            F_temp.B[a][i]    = F->B[a][i]    + DT * k3.B[a][i];
        }
    }
    compute_derivatives(&F_temp, &k4);
    
    pthread_t threads[NUM_THREADS];
    CombineData threadData[NUM_THREADS];
    int chunk = N / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; t++) {
        threadData[t].start = t * chunk;
        threadData[t].end = (t == NUM_THREADS - 1) ? N : (t + 1) * chunk;
        threadData[t].F = F;
        threadData[t].k1 = &k1;
        threadData[t].k2 = &k2;
        threadData[t].k3 = &k3;
        threadData[t].k4 = &k4;
        pthread_create(&threads[t], NULL, combine_thread, &threadData[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
    
    arena_reset(&tempArena);
}

void print_diagnostics(const Fields *F, int step) {
    double sum_alpha = 0.0;
    double sum_traceK = 0.0;
    double max_traceK = 0.0;
    for (int i = 0; i < N; i++) {
        sum_alpha += F->alpha[i];
        double traceK = F->K[0][0][i] + F->K[1][1][i] + F->K[2][2][i];
        sum_traceK += traceK;
        if (fabs(traceK) > max_traceK)
            max_traceK = fabs(traceK);
    }
    double avg_alpha = sum_alpha / N;
    double avg_traceK = sum_traceK / N;
    printf("Step %d: avg_alpha = %lf, avg_traceK = %lf, max_traceK = %lf\n",
           step, avg_alpha, avg_traceK, max_traceK);
}

int main() {
    Fields F;
    allocate_fields(&F);
    initialize_fields(&F, METRIC_SCHWARZSCHILD);

    double sum_alpha = 0.0;
    for (int i = 0; i < N; i++) {
        sum_alpha += F.alpha[i];
    }
    printf("Initial avg_alpha = %lf\n", sum_alpha/N);

    for (int step = 0; step < STEPS; step++) {
        RK4_step(&F);
        print_diagnostics(&F, step);
    }

    arena_free(&mainArena);
    arena_free(&tempArena);
    return 0;
}
