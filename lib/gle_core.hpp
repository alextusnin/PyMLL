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

struct rhs_gle{
    Int Nphi;
    Doub det, P_th, power,g0;
    MatDoub  LinearJacobian;
    double* Dint;
    double* gain;
    Complex i=1i;
    double buf_re, buf_im;
    fftw_plan plan_direct_2_spectrum;
    fftw_plan plan_spectrum_2_direct;
    fftw_complex *buf_direct, *buf_spectrum;
    
    rhs_gle(Int Nphii, const double* Dinti, const double g0i , const double* Gaini, const double P_thi)
    {
        std::cout<<"Initialization started\n";
        Nphi = Nphii;
        Dint = new (std::nothrow) double[Nphi];
        gain = new (std::nothrow) double[Nphi];
        g0=g0i;
        //DispTerm = new (std::nothrow) double[2*Nphi];
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            Dint[i_phi] = Dinti[i_phi];
            gain[i_phi] = Gaini[i_phi];
        }

        P_th=P_thi;
        std::cout<<"Saturation power is " << P_th<< "\n";
        buf_direct = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        
        plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_FORWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        std::cout<<"Initialization succesfull\n";
    }

    rhs_gle(Int Nphii, const double* Dinti, const double g0i , const double* Gaini, const double P_thi, const double* Jacobiani)
    {
        std::cout<<"Initialization started\n";
        Nphi = Nphii;
        LinearJacobian.assign(2*Nphi,2*Nphi,0.);
        Dint = new (std::nothrow) double[Nphi];
        gain = new (std::nothrow) double[Nphi];
        g0=g0i;
        //DispTerm = new (std::nothrow) double[2*Nphi];
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            Dint[i_phi] = Dinti[i_phi];
            gain[i_phi] = Gaini[i_phi];
        }

        for (int i_phi = 0; i_phi<2*Nphi; i_phi++){
            for (int j_phi= 0; j_phi<2*Nphi; j_phi++){
                LinearJacobian[i_phi][j_phi] = Jacobiani[i_phi*2*Nphi+j_phi];
            }
        }
        P_th=P_thi;
        std::cout<<"Saturation power is " << P_th<< "\n";
        buf_direct = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        
        plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_FORWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        std::cout<<"Initialization succesfull\n";
    }
    ~rhs_gle()
    {
        delete [] Dint;
        delete [] gain;
        free(buf_direct);
        free(buf_spectrum);
        fftw_destroy_plan(plan_direct_2_spectrum);
        fftw_destroy_plan(plan_spectrum_2_direct);
    }
    void operator() (const Doub x, VecDoub_I &y, VecDoub &dydx) {
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_direct[i_phi][0] = y[i_phi];
            buf_direct[i_phi][1] = y[i_phi+Nphi];
        }
        fftw_execute(plan_direct_2_spectrum);
        power=0;
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            power+= buf_spectrum[i_phi][1]*buf_spectrum[i_phi][1] + buf_spectrum[i_phi][0]*buf_spectrum[i_phi][0];
        }
        power=power;///Nphi/Nphi;
//      std::cout<< power << " ";
//      std::this_thread::sleep_for(3ms);
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_re = Dint[i_phi]*buf_spectrum[i_phi][1] + gain[i_phi]*buf_spectrum[i_phi][0]*1/(1+power/P_th) ;
            buf_im =  -Dint[i_phi]*buf_spectrum[i_phi][0] + gain[i_phi]*buf_spectrum[i_phi][1]*1/(1+power/P_th) ;
            buf_spectrum[i_phi][0]= buf_re; 
            buf_spectrum[i_phi][1]= buf_im;
        }
        fftw_execute(plan_spectrum_2_direct);

        for (int i_phi = 0; i_phi<Nphi; i_phi++){

            dydx[i_phi] = -y[i_phi]+ buf_direct[i_phi][0]/Nphi  - g0*(y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi+Nphi]; 
            dydx[i_phi+Nphi] = -y[i_phi+Nphi]  +  buf_direct[i_phi][1]/Nphi+g0*(y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi];

        }
    }

    void jacobian(const Doub x, VecDoub_I &y, VecDoub_O &dfdx, MatDoub_O &dfdy) {
        dfdy = LinearJacobian;
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            dfdx[i_phi]=0.;
            dfdx[i_phi+Nphi]=0.;
            dfdy[i_phi][i_phi]-= 2*g0*y[i_phi]*y[i_phi+Nphi];
            dfdy[i_phi][i_phi+Nphi] -= g0*(y[i_phi]*y[i_phi] + 3*y[i_phi+Nphi]*y[i_phi+Nphi] );
            dfdy[i_phi+Nphi][i_phi+Nphi] +=  2*g0*y[i_phi]*y[i_phi+Nphi];
            dfdy[i_phi+Nphi][i_phi] +=  g0*(3*y[i_phi]*y[i_phi] + y[i_phi+Nphi]*y[i_phi+Nphi] );

//          for (int j_phi=0; j_phi<Nphi; j_phi++){
//              dfdy[i_phi][j_phi] = LinearJacobian[i_phi][j_phi] - 2*g0*y[i_phi]*y[i_phi+Nphi];
//              dfdy[i_phi][j_phi+Nphi] = LinearJacobian[i_phi][j_phi+Nphi] - g0*(y[i_phi]*y[i_phi] + 3*y[i_phi+Nphi]*y[i_phi+Nphi] );
//              dfdy[i_phi+Nphi][j_phi+Nphi] = LinearJacobian[i_phi+Nphi][j_phi+Nphi] + 2*g0*y[i_phi]*y[i_phi+Nphi];
//              dfdy[i_phi+Nphi][j_phi] = LinearJacobian[i_phi+Nphi][j_phi] + g0*(3*y[i_phi]*y[i_phi] + y[i_phi+Nphi]*y[i_phi+Nphi] );

//          }


        }
        
        
    }
};
void printProgress (double percentage);
std::complex<double>* WhiteNoise(const double amp, const int Nphi);

void* Propagate_SAM(double* In_val_RE, double* In_val_IM,  const double *phi, const double* Dint, const double g0, const double* Gain, const double P_th, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);
void* Propagate_Stiff(double* In_val_RE, double* In_val_IM,  const double *phi, const double* Dint, const double g0, const double* Gain, const double* Jacobian, const double P_th, const int Ndet, const int Nt, const double dt, const double Tmax, const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);


#ifdef  __cplusplus
}
#endif

#endif  
