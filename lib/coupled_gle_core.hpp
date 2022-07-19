#ifndef _MLL_CORE_HPP_
#define _MLL_CORE_HPP_
#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <random>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include <thread>

#include "./../../NR/NR_C301/code/nr3.h"
#include "./../../NR/NR_C301/code/ludcmp.h"
#include "./../../NR/NR_C301/code/stepper.h"
#include "./../../NR/NR_C301/code/stepperdopr853.h"
#include "./../../NR/NR_C301/code/stepperross.h"
#include "./../../NR/NR_C301/code/steppersie.h"
#include "./../../NR/NR_C301/code/odeint.h"



#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#ifdef  __cplusplus
extern "C" {
#endif

struct rhs_coupled_gle{
    Int Nphi;
    Doub det, P_th, power1, power2, g0, J;
    double* Dint1, *Dint2;
    double* kappa1, *kappa2;
    double* gain1, *gain2;
    Complex i=1i;
    double buf_re, buf_im, dphi;
    fftw_plan plan_direct_2_spectrum1, plan_direct_2_spectrum2;
    fftw_plan plan_spectrum_2_direct1, plan_spectrum_2_direct2;
    fftw_complex *buf_direct1, *buf_spectrum1, *buf_direct2, *buf_spectrum2;
    
    rhs_coupled_gle(Int Nphii, const double* Dint1i, const double* Dint2i, const double* kappa1i, const double* kappa2i  , const double Ji,  const double g0i , const double* Gain1i, const double* Gain2i, const double P_thi)
    {
        std::cout<<"Initialization started\n";
        Nphi = Nphii;
        Dint1 = new (std::nothrow) double[Nphi];
        Dint2 = new (std::nothrow) double[Nphi];
        kappa1 = new (std::nothrow) double[Nphi];
        kappa2 = new (std::nothrow) double[Nphi];
        gain1 = new (std::nothrow) double[Nphi];
        gain2 = new (std::nothrow) double[Nphi];
        g0=g0i;
        J = Ji;
        //DispTerm = new (std::nothrow) double[2*Nphi];
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            Dint1[i_phi] = Dint1i[i_phi];
            Dint2[i_phi] = Dint2i[i_phi];
            gain1[i_phi] = Gain1i[i_phi];
            gain2[i_phi] = Gain2i[i_phi];
            kappa1[i_phi] = kappa1i[i_phi];
            kappa2[i_phi] = kappa2i[i_phi];
        }

        P_th=P_thi;
        std::cout<<"Saturation power is " << P_th<< "\n";
        buf_direct1 = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum1 = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        
        buf_direct2 = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum2 = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        
        plan_direct_2_spectrum1 = fftw_plan_dft_1d(Nphi, buf_direct1,buf_spectrum1, FFTW_FORWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct1 = fftw_plan_dft_1d(Nphi, buf_spectrum1,buf_direct1, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        
        plan_direct_2_spectrum2 = fftw_plan_dft_1d(Nphi, buf_direct2,buf_spectrum2, FFTW_FORWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct2 = fftw_plan_dft_1d(Nphi, buf_spectrum2,buf_direct2, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        std::cout<<"Initialization succesfull\n";
    }

    ~rhs_coupled_gle()
    {
        delete [] Dint1;
        delete [] Dint2;
        delete [] gain1;
        delete [] gain2;
        delete [] kappa1;
        delete [] kappa2;
        free(buf_direct1);
        free(buf_spectrum1);
        free(buf_direct2);
        free(buf_spectrum2);
        fftw_destroy_plan(plan_direct_2_spectrum1);
        fftw_destroy_plan(plan_spectrum_2_direct1);
        fftw_destroy_plan(plan_direct_2_spectrum2);
        fftw_destroy_plan(plan_spectrum_2_direct2);
    }
    void operator() (const Doub x, VecDoub_I &y, VecDoub &dydx) {
        //First ring 
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_direct1[i_phi][0] = y[i_phi];
            buf_direct1[i_phi][1] = y[i_phi+Nphi];
        }
        fftw_execute(plan_direct_2_spectrum1);
        power1=0.;
        power2=0.;
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            power1+= buf_spectrum1[i_phi][1]*buf_spectrum1[i_phi][1] + buf_spectrum1[i_phi][0]*buf_spectrum1[i_phi][0];
        }
        power1=power1;///Nphi/Nphi;
//      std::cout<< power << " ";
//      std::this_thread::sleep_for(3ms);
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_re = -kappa1[i_phi]*buf_spectrum1[i_phi][0] + Dint1[i_phi]*buf_spectrum1[i_phi][1] + gain1[i_phi]*buf_spectrum1[i_phi][0]*1/(1+power1/P_th) ;
            buf_im = -kappa1[i_phi]*buf_spectrum1[i_phi][1] -Dint1[i_phi]*buf_spectrum1[i_phi][0] + gain1[i_phi]*buf_spectrum1[i_phi][1]*1/(1+power1/P_th) ;
            buf_spectrum1[i_phi][0]= buf_re; 
            buf_spectrum1[i_phi][1]= buf_im;
        }
        fftw_execute(plan_spectrum_2_direct1);
        
        //Second ring 
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_direct2[i_phi][0] = y[2*Nphi+i_phi];
            buf_direct2[i_phi][1] = y[2*Nphi+i_phi+Nphi];
        }
        fftw_execute(plan_direct_2_spectrum2);
        power2=0;
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            power2+=  buf_spectrum2[i_phi][1]*buf_spectrum2[i_phi][1] + buf_spectrum2[i_phi][0]*buf_spectrum2[i_phi][0];
        }
        power2=power2;///Nphi/Nphi;
//      std::cout<< power << " ";
//      std::this_thread::sleep_for(3ms);
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_re = -kappa2[i_phi]*buf_spectrum2[i_phi][0]   + Dint2[i_phi]*buf_spectrum2[i_phi][1] + gain2[i_phi]*buf_spectrum2[i_phi][0]*1/(1+power2/P_th)  ;
            buf_im = -kappa2[i_phi]*buf_spectrum2[i_phi][1]  -Dint2[i_phi]*buf_spectrum2[i_phi][0] + gain2[i_phi]*buf_spectrum2[i_phi][1]*1/(1+power2/P_th)  ;
            buf_spectrum2[i_phi][0]= buf_re; 
            buf_spectrum2[i_phi][1]= buf_im;
        }
        fftw_execute(plan_spectrum_2_direct2);


        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            dydx[i_phi] =  buf_direct1[i_phi][0]/Nphi  - g0*(y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi+Nphi]  ; 
            dydx[i_phi+Nphi] =    buf_direct1[i_phi][1]/Nphi+g0*(y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi] )*y[i_phi];

            dydx[i_phi]+= -(J)*(y[2*Nphi+i_phi+Nphi]);
            dydx[i_phi+Nphi]+= (J)*(y[2*Nphi+i_phi]);

            dydx[2*Nphi+i_phi] =  buf_direct2[i_phi][0]/Nphi  - g0*(y[2*Nphi+i_phi]*y[2*Nphi+i_phi]+y[2*Nphi+i_phi+Nphi]*y[2*Nphi+i_phi+Nphi])*y[2*Nphi+i_phi+Nphi]; 
            dydx[2*Nphi+i_phi+Nphi] =   buf_direct2[i_phi][1]/Nphi+g0*(y[2*Nphi+i_phi]*y[2*Nphi+i_phi]+y[2*Nphi+i_phi+Nphi]*y[2*Nphi+i_phi+Nphi])*y[2*Nphi+i_phi];

            dydx[2*Nphi+i_phi]+=  -(J)*(y[i_phi+Nphi]);
            dydx[2*Nphi+i_phi+Nphi]+=  (J)*(y[i_phi]);

        }




    }

};
void printProgress (double percentage);
std::complex<double>* WhiteNoise(const double amp, const int Nphi);

void* Propagate_SAM(double* In_val_RE, double* In_val_IM,  const double *phi, const double* Dint1, const double* Dint2, const double* kappa1, const double* kappa2,  const double J, const double g0, const double* Gain1, const double* Gain2,  const double P_th, const int Ndet, const int Nt, const double dt, const double Tstep, const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);


#ifdef  __cplusplus
}
#endif

#endif  
