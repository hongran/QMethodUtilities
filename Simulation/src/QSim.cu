#include "QSim.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


/*
   GPU kernel function to initialize the random states.
Each thread gets same seed, a different sequence number,
and no offset
*/
__global__ void init_rand( curandState *state, unsigned long long offset, unsigned long long seed) {

   // thread index
   int idx = blockIdx.x*256 + threadIdx.x;

   curand_init( seed, idx, 0, &state[idx]);
}

/*
GPU kernel utility function to initialize fill/flush data arrays
*/
__global__ void zero_int_array( int32_t *array, int length) {

   // thread index
  int idx = blockIdx.x*256 + threadIdx.x;

  if (idx < length) *(array + idx) = 0;
}

/*
GPU kernel utility function to initialize fill/flush data arrays
*/
__global__ void zero_float_array( float *array, int length) {

   // thread index
  int idx = blockIdx.x*256 + threadIdx.x;
  if (idx < length) {
    *(array + idx) = 0.0;
  }
}

/*
GPU kernel user function to build uniform time distribution
*/
__global__ void make_rand( curandState *state, float *randArray) {

   // thread index
   int idx = blockIdx.x*256 + threadIdx.x;

   curandState localState = state[idx];
   randArray[idx] = curand_uniform( &localState);
   state[idx] = localState;
}

/*
GPU kernel user function to build decay curve distribution
*/
__global__ void make_randexp( curandState *state, float *randArray, float tau) {

   // thread index
   int idx = blockIdx.x*256 + threadIdx.x;

   curandState localState = state[idx];
   randArray[idx] = -tau * log( 1.0 -curand_uniform(&localState) );
   state[idx] = localState;
}

/*
GPU kernel user function to build each fills time distribution

fills array fillSumArray[] of size fillls per batch * xtals * time bins with ADC values (no threshold) for all fills in given flush

also fills hitSumArray (fake Tmethod), energySumArray (diagnostics)
of size xtals * time bins (they contain all hits, energy, etc in batch)
*/
__global__ void make_randfill( curandState *state, float *pulseTemplate, float *pedestalTemplate, float *hitSumArray, float *fillSumArray,float *fillSumArrayPed,  float *energySumArray, float *analyzedQSumArray, int *SimulationIntParameters, float * SimulationParameters, int * AnalysisIntParameters, float * AnalysisParameters){

  int TemplateSize = SimulationIntParameters[5]; 
  int TemplateZero = SimulationIntParameters[6];
  int NElectronsPerFill = SimulationIntParameters[2]; 
  int fill_buffer_max_length = SimulationIntParameters[1];
  int nFlushesPerBatch = SimulationIntParameters[4];
  int nFillsPerFlush = SimulationIntParameters[3];

  float threshold = AnalysisParameters[0];
  int window = AnalysisIntParameters[1];
  int gap = AnalysisIntParameters[2];
  bool fillnoise = SimulationIntParameters[7]; 
  bool flashgainsag = SimulationIntParameters[8];
  bool realtime_analysis = AnalysisIntParameters[0];

  float noise = SimulationParameters[1];

  // single thread make complete fill with NElectronsPerFill electrons

   const float tau = 6.4e4;                              // muon time-dilated lifetime (ns)
   float omega_a = SimulationParameters[0];                    // muon anomalous precession frequency (rad/ns)
   const float magicgamma = 29.3;                        // gamma factor for magic momentum 3.094 GeV/c
   const float SimToExpCalCnst = 0.057;                  // energy-ADC counts conversion (ADC counts / energy GeV)
   const float qBinSize = TIMEDECIMATION*1000/800;        // Q-method histogram bin size (ns), accounts for 800MHz sampling rate
   const float Elab_max = 3.095;                         // GeV, maximum positron lab energy
   const float Pi = 3.1415926;                           // Pi
   const float cyclotronperiod = 149.0/qBinSize;        // cyclotron period in histogram bin units
   const float anomalousperiod = 4370./qBinSize;        // anomalous period omega_c-omega_s in histogram bin units
   const int nxseg = NXSEG, nyseg = NYSEG, nsegs = NSEG; // calorimeter segmentation

   // parameters for empirical calculation of positron drift time from energy via empirical polynomial
   float p0 =    -0.255134;
   float p1 =      65.3034;
   float p2 =     -705.492;
   float p3 =      5267.21;
   float p4 =     -23986.5;
   float p5 =      68348.1;
   float p6 =      -121761;
   float p7 =       131393;
   float p8 =       -78343;
   float p9 =      19774.1;

   // parameters to mimic first-order pileup in T-method (no spatial resolution of pulse pileup)
   float PUtick, prevtick, prevADC; // mimic T-method pileup
   float PUdt = 10.0/qBinSize; // convert from ns to clock ticks
   float PUth = 1.8*GeVToADC/PeakBinFrac; // convert from GeV to ADC value
   PUdt = 0.0/qBinSize; // switch off PU
   PUth = 0.0*GeVToADC/PeakBinFrac; // switch off PU

   // variables for muon decay / kinematics 
   float y, A, n;   // mu-decay parameters
   double t, rt; // pars for time dists of mu-decays (float->double after seeing non-randomness in decay curve)  
   float r, r_test; // pars for energy dists of mu-decays  

   // thread index
   int idx = blockIdx.x*blockDim.x + threadIdx.x;

   // one thread per fill - max threads is "fills in flush" * "flushes in batch"
//   printf("idx, FlushPerBatch, FillPerFlush: %d,%d,%d\n",idx,nFlushesPerBatch,nFillsPerFlush);
   if (idx < nFlushesPerBatch * nFillsPerFlush) {

     int iflush = idx / nFillsPerFlush; // fills are launched in blocks of nFlushesPerBatch * nFillsPerFlush, iflush is flush index within flush batch
     int ifill = idx % nFillsPerFlush;  // fills are launched in nlocks of nFlushesPerBatch * nFillsPerFlush, ifill is fill index within individual flush

     // state index for random number generator
     curandState localState = state[idx];
     
     // make noise for each fill if fillnoise true (time consuming)
     if (fillnoise) {

       /*
       // simple fixed pedestal, gaussian noise - SWITCHED OFF
       float pedestal = 0., sigma = 4.; // pedestal, sigma parameters for noise distribution
       for (int i = 0; i < nsegs*fill_buffer_max_length; i++){ // loop over each time bin of each xtal
	 int32_t noise = pedestal + sigma * curand_normal(&localState); // generate Gaussian noise using normal distribution
         atomicAdd( &(fillSumArray[ i ]), (float)noise );       // add fill-by-fill noise to flush buffer
       }
       */

       // empirical time-dependent pedestal, gaussian noise
       // for parameter values see ~/Experiments/g-2/Analysis/June2017/tailstats.ods
       float shrt_tau = 14000./qBinSize; // shrt pedestal lifetime ns->ticks 
       float long_tau = 77000./qBinSize; // long pedestal lifetime ns->ticks
       //float shrt_ampl = 9.1; // amplitude of shrt pedestal lifetime at t=0 in ADC counts per single fill, June 2017
       //float long_ampl = 2.7; // amplitude of long pedestal lifetime at t=0 in ADC counts per single fill, June 2017
       float shrt_ampl = 20.; // amplitude of shrt pedestal lifetime at t=0 in ADC counts per single fill, Jan 2018, 8129
       float long_ampl = 2.0; // amplitude of long pedestal lifetime at t=0 in ADC counts per single fill, Jan 2018, 8129
       shrt_ampl = shrt_ampl;
       long_ampl = long_ampl;
       float sigma = 1.0; // parameters of noise at flush level, noise per single fill to noise per nfill flush
       float prodtocom = -1.0; // commissioning to production run normalization factor of pedestal variation per fill (note sign)
       //float prodtocom = -0.17; // commissioning to production run normalization factor of pedestal variation per fill (note sign)
       //float prodtocom = 0.0; // commissioning to production run normalization factor of pedestal variation per fill

	     for (int i = 0; i < nsegs*fill_buffer_max_length; i++){ // loop over each time bin of each xtal (fill_buffer_max_length is Q-method bins per segment per fill)
	       int iseg = i / fill_buffer_max_length; // xtal index
	       int ibin = i % fill_buffer_max_length; // time bin index
	       /*** add pedestal and statistical noise at flush level ***/
	       float pedestal = prodtocom * ( shrt_ampl*exp(-float(ibin)/shrt_tau) + long_ampl*exp(-float(ibin)/long_tau) ); // xtal independent 
	       //float noise = pedestal + sigma * curand_normal(&localState); // pedestal w/ noise
	       //state[ nFillsPerBatch*nsegs*fill_buffer_max_length + i] = localState; // pedestal w/ noise
	       float noise = pedestal; // pedestal w/o noise
//	       atomicAdd( &(fillSumArray[ iflush*nsegs*fill_buffer_max_length + i ]), noise ); // fill with noise for particular fill in batch
//	       atomicAdd( &(fillSumArrayPed[ iflush*nsegs*fill_buffer_max_length + i ]), noise ); // fill with noise for particular fill in batch
	     } // end loop over samples * xtals
      } // end if fill-by-fill noise

     int nhit = 0; // hit counter
     float theta = 0; // decay angle
     double tick, rtick;    
     double tickstore[NEMAX]; 
     float xrand, yrand, xmax; // x,y coordinate random numbers and endpoint of hit distribution across calo     
     float ylab, phase, drifttime; // parameters for calculating the positron drift time
     float  ADC, xcoord, ycoord; // paramet ters for positron time, ADC counts, and x/y coordinates
     float ADCstore[NEMAX], xcoordstore[NEMAX], ycoordstore[NEMAX]; // arrays for storing hit info before time-ordering (ADCnorm is used for pile-up correction)
     int iold[NEMAX]; // array for storing hit time-ordering 

     // find hit times, energies, x,y-coordinates for ne generated electrons from muon decay
     while (nhit < NElectronsPerFill){ // should randomize the hits per fill

       // get muon decay time, tick, from time-dilted exponential decay
       
       rt = curand_uniform_double(&localState);  

       t = -tau * log( 1.0 - rt );     // random from exp(-t/tau) using uniform random number 0->1
       tick = t/qBinSize;                                      // convert from ns to Q-method histogram bin units
       int itick = (int)tick; // q-method histogram bin from decay time in bin units
       if ( itick  >= fill_buffer_max_length ) continue; // time out-of-bounds 
     
       // get positron lab energy, Elab, by generating the positron energy, angle distribution in muon rest frame
       y = curand_uniform(&localState);
       A = (-8.0*y*y + y + 1.)/( y*y - 5.*y - 5.);
       n = (y - 1)*( y*y - 5.*y - 5.);
       r_test = n*(1.-A*cos(omega_a*t))/6.; 
       r = curand_uniform(&localState);  
       if ( r >= r_test ) continue;
       float Elab = Elab_max*y;

       // if using hitSumArray flush buffer
       atomicAdd( &(hitSumArray[ itick ]), 1.0);
       //if ( ADC > PUth ) atomicAdd( &(hitSumArray[ itick ]), 1.0);

       // account for acceptance of calorimeter using empirical, energy-dependent calo acceptance 

       // very simple empirical acceptance, zero below ElabMin, unit above ElabMin
       //float ElabMin = 0.5; // acceptance cutoff
       //if (Elab < ElabMin) continue;

       // simple acceptance paramterization from Aaron see Experiments/g-2/Analysis/Qmethod/functionForTim.C
       if (Elab > 0.7*Elab_max) {
	       r_test = 1.0;
       } else {
	       r_test = Elab/(0.7*Elab_max);
       }
       r = curand_uniform(&localState);  
       if ( r >= r_test ) continue;

       // variable ADC is total ADC samples of positron signal at 800 MMz sampling rate with 6.2 GeV max range over 2048 ADC counts
       ADC = GeVToADC*Elab; 
       // divide by maximum fraction of positron signal in single 800 MHz bin (is ~0.4 from erfunc plot of 5ns FWHM pulse in peak sample at 800 MHz sampling rate)
       //TODO: Will check this issue back later
       //ADC = ADC/PeakBinFrac; 

       // add empirical energy-dependent drift time, see https://muon.npl.washington.edu/elog/g2/Simulation/229 using empirical energy and time relation
       ylab = Elab/Elab_max;
       phase = p0 + p1*ylab + p2*ylab*ylab + p3*ylab*ylab*ylab + p4*ylab*ylab*ylab*ylab 
	 + p5*ylab*ylab*ylab*ylab*ylab + p6*ylab*ylab*ylab*ylab*ylab*ylab + p7*ylab*ylab*ylab*ylab*ylab*ylab*ylab 
	 + p8*ylab*ylab*ylab*ylab*ylab*ylab*ylab*ylab + p9*ylab*ylab*ylab*ylab*ylab*ylab*ylab*ylab*ylab; // phase in mSR units of omega_a, max is 5.14662666320800781e+01 msr
       drifttime = anomalousperiod * phase / (2.*Pi*1000.); // convert the omega_a phase to drift time in Q-method histogram bin units
       /*
       tick = tick + drifttime; // add drift time to decay time
       itick = (int)tick;
       //do this later
*/
       // if using energySumArray flush buffer
//       atomicAdd( &(energySumArray[ (int)(ADC/TIMEDECIMATION) ]), 1.0); // diagnostic for energy time distributions
   
       // generate the x, y coordinates of positron hit on calorimeter

       // very simple random (x, y) coordinates
       //xcoord = nxseg * curand_uniform(&localState);
       //ycoord = nyseg * curand_uniform(&localState);
       
       // rough empirical x-distribution obtained from  https://muon.npl.washington.edu/elog/g2/Simulation/258 (Robin)
       // rough empirical y-distribution obtained from  https://muon.npl.washington.edu/elog/g2/Simulation/256 (Pete)
       if ( ylab > 0.7 ) {
	       xmax = 185.-533.3*(ylab-0.7);
       } else {
	       xmax = 185.;
       }
       xrand = curand_uniform(&localState);
       xcoord = xmax*xrand/25.0; // x-coordinate -> segment/xtal unitsNFlushes
       // put hit information (time, ADC counts, x/y coordinates, hit index) into hit array needed for applying pile-up effects
       tickstore[nhit] = tick;
       ADCstore[nhit] = ADC;
       xcoordstore[nhit] = xcoord;
       ycoordstore[nhit] = ycoord;
       iold[nhit] = nhit;

       nhit++; // hit counter
     }

     // we've now got the arrays of time (tick, itick), lab energy / ADC value (Elab / ADC), x/y coordinates (xcoord / ycoord) of positron with empirical acceptance, (x,y) distributions

     // sort array of positron hits into ascending time-order
     int itemp;
     float ftemp;
     //TODO: deal with pile up study later
//     thrust::sort(thrust::seq, tickstore, tickstore+nhit);
     /*
     for (int i = 0; i < nhit; ++i) {
       for (int j = i + 1; j < nhit; ++j) {
	       if (tickstore[i] > tickstore[j]) { // if higher index array element j is earlier (t_j < t_i) than lower index  array element i then swap elements
	         ftemp = tickstore[i]; // swap times if hit i is later than hit j
	         tickstore[i] = tickstore[j];
	         tickstore[j] = ftemp;
	         itemp = iold[i]; // swap indexes if hit i is later than hit j
	         iold[i] = iold[j];
	         iold[j] = itemp;
	       }
       }
     }
     */
     //Clear the fill-by-fill array

     for (int ix = 0; ix < nxseg; ix++) {
       for (int iy = 0; iy < nyseg; iy++) { 	   // calc energy in segment (assume aGaussian distribution about xc, yc)
	 // array offset needed for storing xtal hits in samples array
	 int xysegmentoffset = (ix+iy*nxseg)*fill_buffer_max_length; 

	 for (int k=0; k<fill_buffer_max_length; k++) {
	   fillSumArrayPed[ iflush*nsegs*fill_buffer_max_length + xysegmentoffset + k ] = 0.0; 
	 }

       } // end of y-distribution loop
     } // end of x-distribution loop

     // parameters for empirical Gaussian distribution of energy across neighboring segments. 

     // https://muon.npl.washington.edu/elog/g2/SLAC+Test+Beam+2016/260 and position where energy in 
     // neighboring xtal is 16% (1 sigma) - giving sigma = 0.19 in units of crystal size 
     //float xsig = 0.01, ysig = 0.01; // test with very small spread
     //float xsig = 0.5, ysig = 0.5; // test with very large spread
     float xsig = 0.19, ysig = 0.19; // stanard  distribtion, xtal size units
     
     // parameters for distributing the ADC counts over time bins of q-method histogram 

     // approx sigma width of 2.1ns from https://muon.npl.washington.edu/elog/g2/SLAC+Test+Beam+2016/38
     //float width = 2.1/qBinSize; // pulse sigma in q-method bin width units

     // parameters for pile-up effects 

     // simple time constant, pulse amplitude and normalization parameter of pileup effect of prior pulses
     //OBSOLETE 
     //float tauG = 30.0/qBinSize, ampG = 0.04, ADCnorm = 812;

     float ADCstoresegment[54][NEMAX]; // array used for xtal-by-xtal pileup effects 
     
     for (int i = 0; i < nhit; i++){      // loop over time-ordered positron hits
      
       tick = tickstore[i]; // time array is already time-ordered
       ADC = ADCstore[i]; // ADC, x, y arrays aren't already time-ordered
       xcoord = xcoordstore[i];
       ycoord = ycoordstore[i];
       /*
       ADC = ADCstore[iold[i]]; // ADC, x, y arrays aren't already time-ordered
       xcoord = xcoordstore[iold[i]];
       ycoord = ycoordstore[iold[i]];
       */
       //printf("x, y %f, %f, ADC %f, tick %f\n", xcoord, ycoord, ADC, tick); // debugging

	//  add effects of injection flash via time-dependent gain sag 
	//  with amplitude of ampFlashGainSag and time constant tauFlashGainSag
        //  to all calo pulses - see https://gm2-docdb.fnal.gov/cgi-bin/private/RetrieveFile?docid=10818
       //   amplitude of 0.10 is trypical exptrapolation to time t=0 of hot calo gain sag from beam flash

       float ampFlashGainSag = 0.10, tauFlashGainSag = 1.0e5/qBinSize;
       if (flashgainsag) ADC = ADC * (1.0 - ampFlashGainSag * exp( -tick / tauFlashGainSag) ); 
 
       // need Qmethod bin and time within bin for distributing signal over bins
       // itick is bin of q-method time histogram
       // rtick is time within bin of q-method time histogram
       int itick = (int) tick;
       float rtick = tick - itick;

      //Time distribution
      float TimeBinFractions[5];
      int TickEdgeIndex = rtick*10 - TemplateZero; 
      /*
      //Try first delta shape pulse
      TimeBinFractions[0] = 0.0;
      TimeBinFractions[1] = 1.0;
      for (int tempIdx=2;tempIdx<5;tempIdx++)
      {
        TimeBinFractions[tempIdx] = 0.0;
      }*/
      
      if (TickEdgeIndex > 0)
      {
        TimeBinFractions[0] = pulseTemplate[TickEdgeIndex];
      }else{
        TimeBinFractions[0] = 0.0;
      }
      for (int tempIdx=1;tempIdx<5;tempIdx++)
      {
        int leftIdx = TickEdgeIndex + (tempIdx-1)*qBinSize*10;
        int rightIdx = TickEdgeIndex + tempIdx*qBinSize*10;
        if (leftIdx<0)
        {
          leftIdx = 0;
        }
        if (leftIdx >= TemplateSize)
        {
          leftIdx = TemplateSize-1;
        }
        if (rightIdx >= TemplateSize)
        {
          rightIdx = TemplateSize-1;
        }
        TimeBinFractions[tempIdx] = pulseTemplate[rightIdx] - pulseTemplate[leftIdx];
      }

       // we're now going to distribute the ADC signal accross x,y segments and time bins filling the array fillSumArray of size time bins * xsegs * ysegs

       // loop over the array of xtals and distribute the total ADC counts (ADC) to each xtal (ADC segment) using the hit coordinates xcoord, ycoords and spreads xsig, ysig. 
      //Calculating the gaussian integral for each x or y segments
       float fsegmentsum = 0.0; // diagnostic parameter for distribution of energy over segments
       float SegmentX[NXSEG];
       float SegmentY[NYSEG];
       for (int ix = 0; ix < nxseg; ix++) {
           SegmentX[ix] = 0.5*(-erfcf((ix+1.0-xcoord)/(sqrt(2.)*xsig))+erfcf((ix-xcoord)/(sqrt(2.)*xsig)));
       }
       for (int iy = 0; iy < nyseg; iy++) {
           SegmentY[iy] = 0.5*(-erfcf((iy+1.0-xcoord)/(sqrt(2.)*ysig))+erfcf((iy-ycoord)/(sqrt(2.)*ysig)));
       }
       for (int ix = 0; ix < nxseg; ix++) {
	       for (int iy = 0; iy < nyseg; iy++) { 	   // calc energy in segment (assume aGaussian distribution about xc, yc)
//           float fsegmentx = 0.5*(-erfcf((ix+1.0-xcoord)/(sqrt(2.)*xsig))+erfcf((ix-xcoord)/(sqrt(2.)*xsig)));
//	         float fsegmenty = 0.5*(-erfcf((iy+1.0-ycoord)/(sqrt(2.)*ysig))+erfcf((iy-ycoord)/(sqrt(2.)*ysig)));
           float fsegment = SegmentX[ix]*SegmentY[iy];
	         float ADCsegment = fsegment*ADC;
           fsegmentsum += fsegment;
//	         if (ADCsegment < 1.0) continue; // avoid pileup calc if signal in xtal is neglibible
           
	         // array offset needed for storing xtal hits in samples array
	         int xysegmentoffset = (ix+iy*nxseg)*fill_buffer_max_length; 
     
           //Time distribution
           for (int k=0; k<5; k++) {
             int kk = k + itick -1;
             if ( kk < 0 || kk >= fill_buffer_max_length ) continue;
             float ADCfrac = ADCsegment*TimeBinFractions[k]; 

//	     if ( ADCfrac >= 1 ) {
	       //printf("xtal %i, ADCfrac %f\n", ix+iy*nxseg, ADCfrac);
	       atomicAdd( &(fillSumArray[ iflush*nsegs*fill_buffer_max_length + xysegmentoffset + kk ]), ADCfrac/TIMEDECIMATION); // divide by time decimation to account for data processing calculates the time-decimated average not sum
	       fillSumArrayPed[ iflush*nsegs*fill_buffer_max_length + xysegmentoffset + kk ] = ADCfrac/TIMEDECIMATION; // Keep this array only to this fill and use it for analysis, add pedestal later
//	     }
	   }
           //OBSOLETE do time smearing of positron pulse over several contiguous time bins, just loop over bins k-1, k, k+1 as negligible in other bins   
           /*
	         float tfracsum = 0.0; // diagnostic for distribution of energy over segments
	         for (int k=-1; k<=1; k++) {
	           int kk = k + itick;
	           if ( kk < 0 || kk >= fill_buffer_max_length ) continue;
	           float tfrac = 0.5*(-erfcf((kk+1.0-tick)/(sqrt(2.)*width))+erfcf((kk-tick)/(sqrt(2.)*width))); // energy in bin (assume a Gaussian distribution about tick (time within central bin)
             float ADCfrac = ADCsegment*tfrac; 
	            tfracsum += tfrac;

             // FIXME overflow below is incorrect when applied on time decimated bins 
	           //if ( ADCfrac > 2048. ) ADCfrac = 2048.; // apply overflow of ADC counts


	           if ( ADCfrac >= 1 ) {
               //printf("xtal %i, ADCfrac %f\n", ix+iy*nxseg, ADCfrac);
	             atomicAdd( &(fillSumArray[ iflush*nsegs*fill_buffer_max_length + xysegmentoffset + kk ]), ADCfrac/TIMEDECIMATION); // divide by time decimation to account for data processing calculates the time-decimated average not sum
	            }

             } // end of time smearing
            */

           // for no time smearing all xtal ADC counts in single time bin
	     //atomicAdd( &(fillSumArray[ xysegmentoffset + itick ]), (float)ADCsegment );

	         } // end of y-distribution loop
        } // end of x-distribution loop

     } // end of time-ordered hits hits

     //Add pedestal
     for (int ix = 0; ix < nxseg; ix++) {
       for (int iy = 0; iy < nyseg; iy++) { 	   // calc energy in segment (assume aGaussian distribution about xc, yc)
	 // array offset needed for storing xtal hits in samples array
	 int xysegmentoffset = (ix+iy*nxseg)*fill_buffer_max_length; 

	 for (int k=35; k<fill_buffer_max_length; k++) {
	   ;
	   atomicAdd( &(fillSumArrayPed[ iflush*nsegs*fill_buffer_max_length + xysegmentoffset + k ]), pedestalTemplate[k]); 
	 }

       } // end of y-distribution loop
     } // end of x-distribution loop

     // state index for random number generator
     state[idx] =  localState;
     
     //atomicAdd( &(fillSumArray[ iflush*nsegs*fill_buffer_max_length + 100 ]), 999. ); // debuging

     //Analysis, based on fillSumArrayPed, which is updated per fill
     for (int ix = 0; ix < nxseg; ix++) {
       for (int iy = 0; iy < nyseg; iy++) { 	
	 for (int i = 0; i < nhit; i++){      // loop over time-ordered positron hits
	   int xysegmentoffset = (ix+iy*nxseg)*fill_buffer_max_length; 
	   tick = tickstore[i]; // time array is already time-ordered
	   int itick = (int) tick;
           if (itick < window+gap || itick >= fill_buffer_max_length-window-gap) 
	   {
	     continue;
	   }
	   float avg = 0.0;
           for (int k=0; k<window; k++) 
	   {
	     avg += fillSumArrayPed[ iflush*nsegs*fill_buffer_max_length + xysegmentoffset + itick - gap - k ];
	     avg += fillSumArrayPed[ iflush*nsegs*fill_buffer_max_length + xysegmentoffset + itick + gap + k ];
	   }
	   avg /= (2.0*window);

	   atomicAdd( &(analyzedQSumArray[ iflush*nsegs*fill_buffer_max_length + xysegmentoffset + itick ]), fillSumArrayPed[ iflush*nsegs*fill_buffer_max_length + xysegmentoffset + itick ] - avg); 
	 }
       }
     }

   } // end of if idx < nFillsPerFlush
}

/*
GPU kernel function - builds fillSumArray from fillArray if fillArray is used and introduces noise at flush-level
*/
__global__ void make_flushbatchsum( curandState *state, float *fillSumArray, float *fillSumArrayPed, float *batchSumArray, float *batchSumArrayR,float *batchSumArrayErr, float *batchSumArrayRErr, float *PUhitSumArray, int nFillsPerBatch, int nFillsPerFlush, float threshold, int window, int fill_buffer_max_length) {

  // thread index
  int idx = blockIdx.x*256 + threadIdx.x;
  curandState localState = state[idx];

  int nxsegs = NXSEG, nysegs = NYSEG, nsegs = NSEG;

  int kmin = window, kmax = window; // bookending parameters for rolling average calc

  int shift_puL = 2, shift_puH = 2; // location of pileup window histograms

  // fill_buffer_max_length is Q-method bins per segment per fill
  if (idx < nsegs*fill_buffer_max_length ) {
      
    // add all the flushes in batch into batchSumArray
    for (int iflsh = 0; iflsh < nFillsPerBatch; iflsh++) {

      //*(batchSumArray + idx) += *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + idx);  // fill buffer
      if ( (idx%fill_buffer_max_length >= kmin+shift_puL) && (idx%fill_buffer_max_length <= fill_buffer_max_length-kmax-shift_puH-1) ) { // do bookending of histogramming that's needed for rolling average
	
	double ycalcped = 0.0, ycalcsig = 0.0, ysum = 0.0, ysum2 = 0.0, ydiff = 0.0, ydiff_puL =0.0, ydiff_puH =0.0, yorig = 0.0, yorig_puL = 0.0, yorig_puH = 0.0, ytrueped = 0.0, ytruediff = 0.0; // other parameters in rolling average calc
        

	yorig = *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + idx); // difference of ith sample from zero (fixed threshold)
	ytrueped = *(fillSumArrayPed + iflsh*nsegs*fill_buffer_max_length + idx); // true ith sample pedestal
	yorig_puL = *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + idx - shift_puL); // difference of ith sample from zero (fixed threshold)
	yorig_puH = *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + idx - shift_puH); // difference of ith sample from zero (fixed threshold)

        int k = 0;
	// calculate mean of samples in rolling window of -kmin to +kmax around sample i
	for (k = idx-kmin; k <= idx+kmax; k++) {
	  if ( k != idx ) ysum += *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + k);  // average value in rolling window
	}
	ycalcped = ysum / ( kmin + kmax );
	// calculate variance of samples in rolling window of -kmin to +kmax around sample i
	for (k = idx-kmin; k <= idx+kmax; k++) {                                                                                                                                                   	  
	  if ( k != idx ) ysum2 +=  ( *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + k) - ycalcped) * ( *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + k) - ycalcped);  // variance in rolling window                                                                    
	}                                                                                                                                                                                         
	ycalcsig = sqrt( ysum2 / ( kmin + kmax ) );  
 

        int ndrop = 0; // store in dropsamples[] and count in ndrop the pedestal samples too far from pedestal average 
        bool dropsample[33] = { false };
        double chauvenetcoeff = 2.0;        
	for (k = idx-kmin; k <= idx+kmax; k++) {
	  if ( k != idx ) { 
	    if ( abs( *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + k) - ycalcped ) / ycalcsig > chauvenetcoeff ) {                         
	      dropsample[k-idx+kmin] = true;                                                                                                                                                       
	      ndrop++;
	      //if ( idx%fill_buffer_max_length >= 1000 && idx%fill_buffer_max_length <= 1100 ) printf( "signal sample %i, rejected pedestal sample %i, ADC, pedestal %f, %f deviation %f\n", 
	      //      idx, k,  *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + k), ycalcped, abs( *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + k) - ycalcped ) / ycalcsig );
	    } // identify, count rejected points 
	  }
        }
	
        ysum = 0.0; // eliminate the dropped samples from pedestal calculation
	for (k = idx-kmin; k <= idx+kmax; k++) {
	  if ( k != idx && !dropsample[k-idx+kmin] ) ysum += *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + k);  // average value in rolling window
	}
	ycalcped = ysum / ( kmin + kmax - ndrop );
	//if ( idx%fill_buffer_max_length >= 1000 && idx%fill_buffer_max_length <= 1100 && ndrop > 0) printf("sample %i, ndrop %i, old pedestal %f\n", idx, ndrop, ycalcped);

	//if (idx%fill_buffer_max_length == 3000  && yorig >= threshold) {
	//  printf("center time bin, window time bin, ysum, %i, %i, %f\n", 
	//	   idx%fill_buffer_max_length, k, ysum);
	//}

	//if (idx%fill_buffer_max_length == 3000  && yorig >= threshold) {
	//  printf("\ncenter time bin, window time bin, ycalcped, %i, %i, %f\n\n", 
	//	 idx%fill_buffer_max_length, k, ycalcped);
	//}

	// search for samples not close to mean  
	//for (unsigned int k = idx-kmin; k <= idx+kmax; k++) if ( k != idx ) dy = ycalcped - *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + k) / ( kmin + kmax );

        ydiff = yorig - ycalcped; // signal relative to calculated pedestal
        ytruediff = yorig - ytrueped; // signal relative to true pedestal
	ydiff_puL =yorig_puL - ycalcped; // pileup signal relative to calculated pedestal
	ydiff_puH =yorig_puH - ycalcped; // pileup signal relative to calculated pedestal

        //if (ycalcped != 0.0) printf("idx %i, ycalcped %f, ytrueped %f\n", idx, ycalcped, ytrueped);
        //if (ydiff > threshold ) printf("--- xtal, sample, thres, y, ycalcped, ysum %i, %i, %f, %f, %f %f ---\n", 
	//   idx/fill_buffer_max_length, idx%fill_buffer_max_length, threshold, ydiff, ycalcped, ysum ); // debugging
        //if (ydiff > threshold && (ydiff- ycalcped) < threshold ) printf("*** xtal, sample, thres, y, ycalcped, ysum %i, %i, %f, %f, %f %f ***\n", 
	//   idx/fill_buffer_max_length, idx%fill_buffer_max_length, threshold, ydiff, ycalcped, ysum ); // debugging

	/* fixed threshold data
	if ( yorig > threshold) {
	  *(batchSumArray + idx) += yorig;  // fill buffer, wothout rolling threshold
	  *(batchSumArrayErr + idx) += sqrt( *(batchSumArrayErr + idx) * *(batchSumArrayErr + idx) + yorig);  // fill buffer, without calculated pedestal
          //printf("fill fixed threshold array: sample, yped, yorig, ydiff %i, %f, %f, %f\n", idx%fill_buffer_max_length, ycalcped, yorig, ydiff);
	}
	*/
     
        //if (idx%fill_buffer_max_length == 3000 && yorig >= threshold) {
	//  printf("time bin, ycalcped, ytrueped, ydiff, ytruediff, yorig %i, %f, %f, %f %f %f", 
	//	 idx%fill_buffer_max_length, ycalcped, ytrueped, ydiff, ytruediff,  yorig);
        //  if (ydiff >= threshold) {
	//    printf("** got hit **\n");
	//  } else {
	//    printf("** lost hit **\n");
	//  }
	//}

        // calculated pedestal data
	if ( ydiff >= threshold ) {
	  *(batchSumArrayR + idx) += ydiff;  // fill buffer, with rolling pedestal
	  //*(batchSumArrayR + idx) += ycalcped;  // fill buffer, with rolling pedestal
	  *(batchSumArrayRErr + idx) = sqrt( *(batchSumArrayRErr + idx) * *(batchSumArrayRErr + idx) + ydiff*ydiff );  // fill buffer, with calculated pedestal
          //printf("fill calculated array: sample, yped, yorig, ydiff %i, %f, %f, %f sum, sumErr %f, %f\n", idx%fill_buffer_max_length, ycalcped, yorig, ydiff, *(batchSumArrayR + idx), *(batchSumArrayRErr + idx) );
	  if ( ydiff_puL > ycalcsig*chauvenetcoeff && ydiff_puL < threshold ) {
	    *(PUhitSumArray + idx%fill_buffer_max_length) += ydiff_puL;
	    //printf("pileup hit, time bin, ycalcped, ytrueped, ydiff, ytruediff, yorig %i, %f, %f, %f %f %f, yorig_puL %f\n", 
	    //	 idx%fill_buffer_max_length, ycalcped, ytrueped, ydiff, ytruediff, yorig, yorig_puL);
	  }
	  if ( ydiff_puH > ycalcsig*chauvenetcoeff && ydiff_puH < threshold ) {
	    *(PUhitSumArray + idx%fill_buffer_max_length) += ydiff_puH;
	    //printf("pileup hit, time bin, ycalcped, ytrueped, ydiff, ytruediff, yorig %i, %f, %f, %f %f %f, yorig_puH %f\n", 
	    //	 idx%fill_buffer_max_length, ycalcped, ytrueped, ydiff, ytruediff, yorig, yorig_puH);
	  }
	}
	
        // true pedestal data
	if ( ytruediff >= threshold ) {
	  *(batchSumArray + idx) += ytruediff;  // fill buffer, with true pedestal
	  //*(batchSumArray + idx) += ytrueped;  // fill buffer, with true pedestal
	  *(batchSumArrayErr + idx) = sqrt( *(batchSumArrayErr + idx) * *(batchSumArrayErr + idx) + yorig*yorig );  // fill buffer, with true pedestal
          //printf("fill true pedestal array: sample, yped, yorig, ydiff %i, %f, %f, %f  sum, sumErr %f, %f\n", idx%fill_buffer_max_length, ytrueped, yorig, ytruediff, *(batchSumArray + idx), *(batchSumArrayErr + idx) );
	}


        //printf("sample index, flush index, flush sample batch sample %i, %f, %f\n", idx, iflsh, *(fillSumArray + iflsh*nsegs*fill_buffer_max_length + idx ), *(batchSumArray + idx) ); // debug
      } // end book-ending

 
    } // end loop over flushes in batch

  } // end check on thread index
}



namespace  QSimulation
{
  QSim::QSim(const std::map<std::string,float> & tFloatParameters, const std::map<std::string,int>& tIntParameters,long long int tSeed)
  {
    //Import Parameters
    FloatParameters = tFloatParameters;
    IntParameters = tIntParameters;

    //Additional Parameters
    IntParameters["FillBufferMaxLength"] = nsPerFill / qBinSize;

    //Init Parameter Arrays
    InitParameters();

    //Allocate Derive memory for parameters

    cudaMalloc( (void **)&d_SimulationParameters, SimulationParameters.size()*sizeof(float));
    cudaMalloc( (void **)&d_SimulationIntParameters, SimulationIntParameters.size()*sizeof(int));
    cudaMalloc( (void **)&d_AnalysisParameters, AnalysisParameters.size()*sizeof(float));
    cudaMalloc( (void **)&d_AnalysisIntParameters, AnalysisIntParameters.size()*sizeof(int));
    
    cudaMemcpy( d_SimulationParameters,&SimulationParameters[0], SimulationParameters.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_SimulationIntParameters,&SimulationIntParameters[0], SimulationIntParameters.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_AnalysisParameters,&AnalysisParameters[0], AnalysisParameters.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_AnalysisIntParameters,&AnalysisIntParameters[0], AnalysisIntParameters.size()*sizeof(float), cudaMemcpyHostToDevice);

    cudaError err;

    //Template
    IntegratedPulseTemplate = std::vector<float>(IntParameters["TemplateSize"]);
    for (int i=IntParameters["TemplateZero"];i<IntParameters["TemplateSize"];i++)
    {
      IntegratedPulseTemplate[i] = 1.0;
    }
    PedestalTemplate = std::vector<float>(IntParameters["FillBufferMaxLength"],0.0);
    
    //Arrays

    int nFillsPerBatch = IntParameters["NFlushesPerBatch"] * IntParameters["NFillsPerFlush"];

    ArraySizes["pulseTemplate"] = IntParameters["TemplateSize"]*sizeof(float); // pulse template array
    //ArraySizes["analysisParameters"] = 100*sizeof(float); // Analysis Parameters, max 100
    ArraySizes["pedestalTemplate"] = IntParameters["FillBufferMaxLength"]*sizeof(float); // pulse template array

    ArraySizes["fillSumArray"] = nFillsPerBatch*nsegs*IntParameters["FillBufferMaxLength"]*sizeof(float); // single fill array
    ArraySizes["fillSumArrayPed"] = nFillsPerBatch*nsegs*IntParameters["FillBufferMaxLength"]*sizeof(float); // single fill pedestal array
    ArraySizes["analyzedQSumArray"] = nFillsPerBatch*nsegs*IntParameters["FillBufferMaxLength"]*sizeof(float); // single fill array analyzed

    ArraySizes["batchSumArray"] = nsegs*IntParameters["FillBufferMaxLength"]*sizeof(float); 
    ArraySizes["batchSumArrayErr"] = nsegs*IntParameters["FillBufferMaxLength"]*sizeof(float); 
    ArraySizes["batchSumArrayR"] = nsegs*IntParameters["FillBufferMaxLength"]*sizeof(float); 
    ArraySizes["batchSumArrayRErr"] = nsegs*IntParameters["FillBufferMaxLength"]*sizeof(float); 
    ArraySizes["hitSumArray"] = IntParameters["FillBufferMaxLength"]*sizeof(float); 
    ArraySizes["PUhitSumArray"] = IntParameters["FillBufferMaxLength"]*sizeof(float); 

    // get some cuda device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device Number: %d\n", 0);
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("\n\n");

    cudaMalloc( (void **)&d_state, nsegs*nFillsPerBatch*sizeof(curandState));

    for (auto it=ArraySizes.begin();it!=ArraySizes.end();++it)
    {
      auto Name = it->first;
      auto Size = it->second;
      HostArrays[Name] = (float *)malloc(Size);
      cudaMalloc( (void **)&DeviceArrays[Name], Size);
    }

    err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure with user kernel function make)randfill %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }

    int nblocks = nFillsPerBatch / IntParameters["NThreadsPerBlock"] + 1;
    long long int rand_seed = tSeed;
    if (rand_seed == 0)
    {
      rand_seed = time(NULL);
    }
    init_rand<<<nblocks,IntParameters["NThreadsPerBlock"]>>>( d_state, 0, rand_seed);

  }

  QSim::~QSim()
  {
    for (auto it=HostArrays.begin();it!=HostArrays.end();++it)
    {
      free(it->second);
    }
    for (auto it=DeviceArrays.begin();it!=DeviceArrays.end();++it)
    {
      cudaFree(it->second);
    }

    cudaFree(d_SimulationParameters);
    cudaFree(d_SimulationIntParameters);
    cudaFree(d_AnalysisParameters);
    cudaFree(d_AnalysisIntParameters);
  }

  int QSim::Simulate(int NFlushes)
  {
    cudaError err;
    int NSim = NFlushes/IntParameters["NFlushesPerBatch"] + 1;
    //Clean device memory
    for (auto it=ArraySizes.begin();it!=ArraySizes.end();++it)
    {
      auto Name = it->first;
      auto Size = it->second;
      cudaMemset( DeviceArrays[Name], 0.0, Size);
    }

    //Load template
    
    cudaMemcpy( DeviceArrays["pulseTemplate"],HostArrays["pulseTemplate"], ArraySizes["pulseTemplate"], cudaMemcpyHostToDevice);
    cudaMemcpy( DeviceArrays["pedestalTemplate"],HostArrays["pedestalTemplate"], ArraySizes["pedestalTemplate"], cudaMemcpyHostToDevice);
    
    int nFillsPerBatch = IntParameters["NFlushesPerBatch"] * IntParameters["NFillsPerFlush"];
    for (int i=0;i<NSim;i++)
    {
      
      //Simulate
      // make the fills
      int nblocks = nFillsPerBatch / IntParameters["NThreadsPerBlock"] + 1;

      //std::cout << nblocks<<std::endl;
      std::cout << "Simulating " << i*IntParameters["NFlushesPerBatch"] <<" flushes "<<std::endl;

      //make_randfill<<<nblocks,nThreadsPerBlock>>>( d_state,DeviceArrays["pulseTemplate"], DeviceArrays["hitSumArray"], DeviceArrays["fillSumArray"], DeviceArrays["fillSumArrayPed"], DeviceArrays["energySumArray"], DeviceArrays["analyzedQSumArray"],DeviceArrays["analysisParameters"],TemplateSize,TemplateZero, nElectronsPerFill, IntParameters["FillBufferMaxLength"], nFillsPerBatch, nFillsPerFlush, threshold, fillnoise, flashgainsag,true);
      make_randfill<<<nblocks,IntParameters["NThreadsPerBlock"]>>>( d_state,DeviceArrays["pulseTemplate"],DeviceArrays["pedestalTemplate"], DeviceArrays["hitSumArray"], DeviceArrays["fillSumArray"], DeviceArrays["fillSumArrayPed"], DeviceArrays["energySumArray"], DeviceArrays["analyzedQSumArray"],d_SimulationIntParameters, d_SimulationParameters,d_AnalysisIntParameters, d_AnalysisParameters);

      cudaDeviceSynchronize();
      err=cudaGetLastError();
      if(err!=cudaSuccess) {
	      printf("Cuda failure with user kernel function make)randfill %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
	      exit(0);
      }
/*
      nblocks = ( nsegs * IntParameters["FillBufferMaxLength"] + nThreadsPerBlock - 1 )/ nThreadsPerBlock;
      make_flushbatchsum<<<nblocks,nThreadsPerBlock>>>( d_state, DeviceArrays["fillSumArray"], DeviceArrays["fillSumArrayPed"], DeviceArrays["batchSumArray"], DeviceArrays["batchSumArrayR"], DeviceArrays["batchSumArrayErr"], DeviceArrays["batchSumArrayRErr"], DeviceArrays["PUhitSumArray"], nFillsPerBatch, nFillsPerFlush, threshold, window, IntParameters["FillBufferMaxLength"]);
      cudaDeviceSynchronize();
      err=cudaGetLastError();
      if(err!=cudaSuccess) {
	      printf("Cuda failure with user kernel function make)randfill %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
	      exit(0);
      }
      */
    }

    //Copy back to host memory
    int n=0;
    for (auto it=ArraySizes.begin();it!=ArraySizes.end();++it)
    {
      auto Name = it->first;
      auto Size = it->second;
      cudaMemcpy( HostArrays[Name], DeviceArrays[Name], Size, cudaMemcpyDeviceToHost);
      std::cout<< n << " "<<Name<<" "<<Size<<std::endl;
      err=cudaGetLastError();
      if(err!=cudaSuccess) {
        printf("Cuda failure with user kernel function make)randfill %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
        exit(0);
      }
      n++;
    }

    return 0;
  }

  int QSim::GetArray(std::string ArrayName,std::vector<double>& Output)
  {
    auto Size = ArraySizes[ArrayName];
    Output.resize(Size);
    auto ptr = HostArrays[ArrayName];
    for (int i=0;i<Size;i++)
    {
      Output[i] = ptr[i];
    }
    return 0;
  }

  int QSim::GetCaloArray(std::string ArrayName,std::vector<double>& Output,bool BatchSum)
  {
    Output.clear();
    Output.resize(IntParameters["FillBufferMaxLength"],0.0);
    auto ptr = HostArrays[ArrayName];

    int nFillsPerBatch = IntParameters["NFlushesPerBatch"] * IntParameters["NFillsPerFlush"];

    for (unsigned int k=0;k<nFillsPerBatch;k++)
    {
      for (unsigned int j=0;j<nsegs;j++)
      {
        for (unsigned int i=0;i<IntParameters["FillBufferMaxLength"];i++)
        {
          Output[i] += ptr[(k*nsegs+j)*IntParameters["FillBufferMaxLength"] + i];
        }
      }
      if (!BatchSum)
      {
	break;
      }
    }
    
    return 0;
  }

  int QSim::SetIntegratedPulseTemplate(std::vector<float> temp,int Size,int ZeroIndex)
  {
    free(HostArrays["pulseTemplate"]);
    cudaFree(DeviceArrays["pulseTemplate"]);

    HostArrays["pulseTemplate"] = (float *)malloc(Size*sizeof(float));
    cudaMalloc( (void **)&DeviceArrays["pulseTemplate"], Size*sizeof(float));

    //TemplateSize = Size;
    //TemplateZero = ZeroIndex;
    IntegratedPulseTemplate = temp;
    IntegratedPulseTemplate.resize(Size);

    for (int i=ZeroIndex;i<Size;i++)
    {
      HostArrays["pulseTemplate"][i] = IntegratedPulseTemplate[i];
    }

    return 0;
  }

  int QSim::SetPedestalTemplate(std::vector<float> temp)
  {
    free(HostArrays["pedestalTemplate"]);
    cudaFree(DeviceArrays["pedestalTemplate"]);
    int Size = temp.size();
    int ReservedSize = ArraySizes["pedestalTemplate"];

    HostArrays["pedestalTemplate"] = (float *)malloc(ReservedSize);
    cudaMalloc( (void **)&DeviceArrays["pulseTemplate"], ReservedSize);

    PedestalTemplate = temp;

    PedestalTemplate.resize(ReservedSize/sizeof(float),0.0);

    if (Size > ReservedSize/sizeof(float))
    {
      Size = ReservedSize;
    }
    for (int i=0;i<Size;i++)
    {
      HostArrays["pedestalTemplate"][i] = PedestalTemplate[i];
    }

    return 0;
  }

  //Private Functions
  int QSim::InitParameters()
  {
    SimulationParameters.resize(2);
    SimulationIntParameters.resize(9);
    AnalysisParameters.resize(1);
    AnalysisIntParameters.resize(3);

    SimulationParameters[0] = FloatParameters["Omega_a"];
    SimulationParameters[1] = FloatParameters["Noise"];

    SimulationIntParameters[0] = IntParameters["NThreadsPerBlock"];
    SimulationIntParameters[1] = IntParameters["FillBufferMaxLength"];
    SimulationIntParameters[2] = IntParameters["NElectronsPerFill"];
    SimulationIntParameters[3] = IntParameters["NFillsPerFlush"];
    SimulationIntParameters[4] = IntParameters["NFlushesPerBatch"]; 
    SimulationIntParameters[5] = IntParameters["TemplateSize"]; 
    SimulationIntParameters[6] = IntParameters["TemplateZero"]; 
    SimulationIntParameters[7] = IntParameters["FillNoiseSwitch"]; 
    SimulationIntParameters[8] = IntParameters["FlashGainSagSwitch"]; 

    AnalysisParameters[0] = FloatParameters["Threshold"];

    AnalysisIntParameters[0] = IntParameters["AnalysisSwitch"];
    AnalysisIntParameters[1] = IntParameters["Window"];
    AnalysisIntParameters[2] = IntParameters["Gap"];
    return 0;
  }

}
