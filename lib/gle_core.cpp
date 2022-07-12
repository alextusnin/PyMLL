#include "gle_core.hpp"

void printProgress (double percentage)
{
    int val = (int) (percentage*100 );
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}
std::complex<double>* WhiteNoise(const double amp, const int Nphi)
{
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::complex<double>* noise_spectrum = new (std::nothrow) std::complex<double>[Nphi];//contains white noise in spectal domain
    std::complex<double>* res = new (std::nothrow) std::complex<double>[Nphi];//contains white noise in spectal domain
    fftw_complex noise_direct[Nphi];
    fftw_plan p;
    
    p = fftw_plan_dft_1d(Nphi, reinterpret_cast<fftw_complex*>(noise_spectrum), noise_direct, FFTW_BACKWARD, FFTW_ESTIMATE);
    double phase;
    double noise_amp;
    const std::complex<double> i(0, 1);
    std::default_random_engine generator(seed1);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (int j=0; j<Nphi; j++){
       phase = distribution(generator) *2*M_PI-M_PI;
       noise_amp  = distribution(generator) *amp;
       noise_spectrum[j] = noise_amp *std::exp(i*phase)/sqrt(Nphi);
    }


    fftw_execute(p);
    for (int j=0; j<Nphi; j++){
        res[j].real(noise_direct[j][0]);
        res[j].imag(noise_direct[j][1]);
    }
    fftw_destroy_plan(p);
    delete [] noise_spectrum;
    return res;
}

void* Propagate_SAM(double* In_val_RE, double* In_val_IM,  const double *phi, const double* Dint, const double g0, const double* gain, const double P_th,  const int Ndet, const int Nt, const double dt, const double Ttotal, const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
    
{
    
    std::cout<<"Pseudo Spectral Step adaptative Dopri853 from NR3 is running\n";
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi];
    VecDoub res_buf(2*Nphi);
    double power = 0.;
    double t0=0., t1=Ttotal;
    double dtmin, dtmax;
    dtmax = t1-t0;
    if (dtmax < dt) {dtmax/=10; dtmin=dtmax/10;}
    else {dtmax = dt; dtmin = dt/10;}
    //std::cout<<"dtmax = " << dtmax << " single step = " << Tstep << " Tmax = " << Tstep*Ndet << "\n";
    std::cout<<"dt = " << dt << " single step = " << Ttotal/Ndet << " Tmax = " << Ttotal << "\n";

    noise=WhiteNoise(noise_amp,Nphi);
    for (int i_phi = 0; i_phi<Nphi; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
        res_buf[i_phi] = res_RE[i_phi] + noise[i_phi].real();
        res_buf[i_phi+Nphi] = res_IM[i_phi] + noise[i_phi].imag();
    }
    Output out(Ndet);
    rhs_gle gle(Nphi, Dint, g0, gain, P_th);
    noise=WhiteNoise(noise_amp,Nphi);
    Odeint<StepperDopr853<rhs_gle> > ode(res_buf,t0,Ttotal,atol,rtol,dt,dt/1000,out,gle);
    ode.integrate();
    std::cout<<"Integration is done, saving data\n";
    for (int i_det=0; i_det<Ndet; i_det++){
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            res_RE[i_det*Nphi+i_phi] = out.ysave[i_phi][i_det];
            res_IM[i_det*Nphi+i_phi] = out.ysave[i_phi+Nphi][i_det];
        }

    }

    delete [] noise;
    std::cout<<"Pseudo Spectral Step adaptative Dopri853 from NR3 is finished\n";
}

void* Propagate_StiffSie(double* In_val_RE, double* In_val_IM,  const double *phi, const double* Dint, const double g0, const double* gain, const double* DispJacobian, const double* GainJacobian, const double P_th,  const int Ndet, const int Nt, const double dt, const double Tstep, const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
    
{
    
    std::cout<<"Pseudo Spectral stiff from NR3 is running\n";
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi];
    //const double t0=0., t1=(Nt-1)*dt, dtmin=0.;
    double t0=0., t1=Tstep;
    double dtmin, dtmax;
    dtmax = t1-t0;
    if (dtmax < dt) {dtmax/=10; dtmin=dtmax/10;}
    else {dtmax = dt; dtmin = dt/10;}
    //std::cout<<"dtmax = " << dtmax << " single step = " << Tstep << " Tmax = " << Tstep*Ndet << "\n";
    std::cout<<"dt = " << dt << " single step = " << Tstep << " Tmax = " << Tstep*Ndet << "\n";

    VecDoub res_buf(2*Nphi);
    double power = 0.;

    noise=WhiteNoise(noise_amp,Nphi);
    for (int i_phi = 0; i_phi<Nphi; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
        res_buf[i_phi] = res_RE[i_phi] + noise[i_phi].real();
        res_buf[i_phi+Nphi] = res_IM[i_phi] + noise[i_phi].imag();
    }
    //std::cout<<"In val_RE = " << In_val_RE[0]<<std::endl;

    Output out(Ndet);
    rhs_gle gle(Nphi, Dint, g0, gain, P_th, DispJacobian, GainJacobian,std::abs(phi[1]-phi[0]) );
    noise=WhiteNoise(noise_amp,Nphi);
    //Odeint<StepperRoss<rhs_gle> > ode(res_buf,t0,Tstep*Ndet,atol,rtol,dt,dt/100,out,gle);
    Odeint<StepperSie<rhs_gle> > ode(res_buf,t0,Tstep*Ndet,atol,rtol,dt,dt/1000,out,gle);
    ode.integrate();
    std::cout<<"Integration is done, saving data\n";
    for (int i_det=0; i_det<Ndet; i_det++){
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            res_RE[i_det*Nphi+i_phi] = out.ysave[i_phi][i_det];
            res_IM[i_det*Nphi+i_phi] = out.ysave[i_phi+Nphi][i_det];
        }

    }
    
    delete [] noise;
    std::cout<<"Pseudo Spectral stiff from NR3 is finished\n";
}


void* Propagate_StiffRoss(double* In_val_RE, double* In_val_IM,  const double *phi, const double* Dint, const double g0, const double* gain, const double* DispJacobian, const double* GainJacobian, const double P_th,  const int Ndet, const int Nt, const double dt, const double Tstep, const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
    
{
    
    std::cout<<"Pseudo Spectral stiff from NR3 is running\n";
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi];
    //const double t0=0., t1=(Nt-1)*dt, dtmin=0.;
    double t0=0., t1=Tstep;
    double dtmin, dtmax;
    dtmax = t1-t0;
    if (dtmax < dt) {dtmax/=10; dtmin=dtmax/10;}
    else {dtmax = dt; dtmin = dt/10;}
    //std::cout<<"dtmax = " << dtmax << " single step = " << Tstep << " Tmax = " << Tstep*Ndet << "\n";
    std::cout<<"dt = " << dt << " single step = " << Tstep << " Tmax = " << Tstep*Ndet << "\n";

    VecDoub res_buf(2*Nphi);
    double power = 0.;

    noise=WhiteNoise(noise_amp,Nphi);
    for (int i_phi = 0; i_phi<Nphi; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
        res_buf[i_phi] = res_RE[i_phi] + noise[i_phi].real();
        res_buf[i_phi+Nphi] = res_IM[i_phi] + noise[i_phi].imag();
    }
    //std::cout<<"In val_RE = " << In_val_RE[0]<<std::endl;

    Output out(Ndet);
    rhs_gle gle(Nphi, Dint, g0, gain, P_th, DispJacobian, GainJacobian,std::abs(phi[1]-phi[0]) );
    noise=WhiteNoise(noise_amp,Nphi);
    Odeint<StepperRoss<rhs_gle> > ode(res_buf,t0,Tstep*Ndet,atol,rtol,dt,dt/1000,out,gle);
    ode.integrate();
    std::cout<<"Integration is done, saving data\n";
    for (int i_det=0; i_det<Ndet; i_det++){
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            res_RE[i_det*Nphi+i_phi] = out.ysave[i_phi][i_det];
            res_IM[i_det*Nphi+i_phi] = out.ysave[i_phi+Nphi][i_det];
        }

    }
    
    delete [] noise;
    std::cout<<"Pseudo Spectral stiff from NR3 is finished\n";
}
