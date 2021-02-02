#include "QSim.h"
#include "TFile.h"
#include "TH1.h"
#include "TSpline.h"
#include "TString.h"
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

/*
   GPU kernel function to initialize the random states.
Each thread gets same seed, a different sequence number,
and no offset
*/
__global__ void init_rand(curandState *state, unsigned long long offset,
                          unsigned long long seed) {

  // thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  curand_init(seed, idx, 0, &state[idx]);
}

/*
GPU kernel utility function to initialize fill/flush data arrays
*/
__global__ void zero_int_array(int32_t *array, int length) {

  // thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < length)
    *(array + idx) = 0;
}

/*
GPU kernel utility function to initialize fill/flush data arrays
*/
__global__ void zero_float_array(float *array, int length) {

  // thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < length) {
    *(array + idx) = 0.0;
  }
}

/*
GPU kernel user function to build uniform time distribution
*/
__global__ void make_rand(curandState *state, float *randArray) {

  // thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandState localState = state[idx];
  randArray[idx] = curand_uniform(&localState);
  state[idx] = localState;
}

/*
GPU kernel user function to build decay curve distribution
*/
__global__ void make_randexp(curandState *state, float *randArray, float tau) {

  // thread index
  int idx = blockIdx.x * 256 + threadIdx.x;

  curandState localState = state[idx];
  randArray[idx] = -tau * log(1.0 - curand_uniform(&localState));
  state[idx] = localState;
}

/*
GPU kernel user function to build each fills time distribution

fills array fillSumArray[] of size fillls per batch * xtals * time bins with ADC
values (no threshold) for all fills in given flush

also fills hitSumArray (fake Tmethod), energySumArray (diagnostics)
of size xtals * time bins (they contain all hits, energy, etc in batch)
*/
__global__ void make_randfill(curandState *state, float *pulseTemplate,
                              float *pedestalTemplate, float *energySumArray,
                              float *flushHitArray, float *flushQTruthArray,
                              float *flushQArray, int *SimulationIntParameters,
                              float *SimulationParameters) {

  int TemplateSize = SimulationIntParameters[5];
  int TemplateZero = SimulationIntParameters[6];
  int NElectronsPerFill = SimulationIntParameters[2];
  int fill_buffer_max_length = SimulationIntParameters[1];
  int nFlushesPerBatch = SimulationIntParameters[4];
  int nFillsPerFlush = SimulationIntParameters[3];

  bool fillnoise = SimulationIntParameters[7];
  bool flashgainsag = SimulationIntParameters[8];

  float noise = SimulationParameters[1];
  float fixedElab = SimulationParameters[2];
  // single thread make complete fill with NElectronsPerFill electrons

  const float tau = 6.4e4; // muon time-dilated lifetime (ns)
  float omega_a =
      SimulationParameters[0];   // muon anomalous precession frequency (rad/ns)
  const float magicgamma = 29.3; // gamma factor for magic momentum 3.094 GeV/c
  const float SimToExpCalCnst =
      0.057; // energy-ADC counts conversion (ADC counts / energy GeV)
  const float qBinSize =
      TIMEDECIMATION * 1000 / 800; // Q-method histogram bin size (ns), accounts
                                   // for 800MHz sampling rate
  const float Elab_max = 3.095;    // GeV, maximum positron lab energy
  const float Pi = 3.1415926;      // Pi
  const float cyclotronperiod =
      149.0 / qBinSize; // cyclotron period in histogram bin units
  const float anomalousperiod =
      4370. /
      qBinSize; // anomalous period omega_c-omega_s in histogram bin units
  const int nxseg = NXSEG, nyseg = NYSEG,
            nsegs = NSEG; // calorimeter segmentation

  // parameters for empirical calculation of positron drift time from energy via
  // empirical polynomial
  float p0 = -0.255134;
  float p1 = 65.3034;
  float p2 = -705.492;
  float p3 = 5267.21;
  float p4 = -23986.5;
  float p5 = 68348.1;
  float p6 = -121761;
  float p7 = 131393;
  float p8 = -78343;
  float p9 = 19774.1;

  // parameters to mimic first-order pileup in T-method (no spatial resolution
  // of pulse pileup)
  float PUtick, prevtick, prevADC;           // mimic T-method pileup
  float PUdt = 10.0 / qBinSize;              // convert from ns to clock ticks
  float PUth = 1.8 * GeVToADC / PeakBinFrac; // convert from GeV to ADC value
  PUdt = 0.0 / qBinSize;                     // switch off PU
  PUth = 0.0 * GeVToADC / PeakBinFrac;       // switch off PU

  // variables for muon decay / kinematics
  float y, A, n; // mu-decay parameters
  double t, rt;  // pars for time dists of mu-decays (float->double after seeing
                 // non-randomness in decay curve)
  float r, r_test; // pars for energy dists of mu-decays

  // thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // one thread per fill - max threads is "fills in flush" * "flushes in batch"
  //   printf("idx, FlushPerBatch, FillPerFlush:
  //   %d,%d,%d\n",idx,nFlushesPerBatch,nFillsPerFlush);
  if (idx < nFlushesPerBatch * nFillsPerFlush) {

    int iflush =
        idx / nFillsPerFlush; // fills are launched in blocks of
                              // nFlushesPerBatch * nFillsPerFlush, iflush is
                              // flush index within flush batch
    int ifill =
        idx % nFillsPerFlush; // fills are launched in nlocks of
                              // nFlushesPerBatch * nFillsPerFlush, ifill is
                              // fill index within individual flush

    // state index for random number generator
    curandState localState = state[idx];

    // make noise for each fill if fillnoise true (time consuming)
    // if (fillnoise) {
    //   ;
    //   // TODO add noise fill by fill
    // } // end if fill-by-fill noise

    int nhit = 0;    // hit counter
    float theta = 0; // decay angle
    double tick, rtick;
    double tickstore[NEMAX];
    float xrand, yrand, xmax; // x,y coordinate random numbers and endpoint of
                              // hit distribution across calo
    float ylab, phase,
        drifttime; // parameters for calculating the positron drift time
    float ADC, xcoord, ycoord; // paramet ters for positron time, ADC counts,
                               // and x/y coordinates
    float ADCstore[NEMAX], xcoordstore[NEMAX],
        ycoordstore[NEMAX]; // arrays for storing hit info before time-ordering
                            // (ADCnorm is used for pile-up correction)
    int iold[NEMAX];        // array for storing hit time-ordering

    // find hit times, energies, x,y-coordinates for ne generated electrons from
    // muon decay
    while (nhit < NElectronsPerFill) { // should randomize the hits per fill

      // get muon decay time, tick, from time-dilted exponential decay

      rt = curand_uniform_double(&localState);

      t = -tau *
          log(1.0 -
              rt); // random from exp(-t/tau) using uniform random number 0->1
      tick = t / qBinSize; // convert from ns to Q-method histogram bin units
      int itick =
          (int)tick; // q-method histogram bin from decay time in bin units
      if (itick >= fill_buffer_max_length)
        continue; // time out-of-bounds

      // get positron lab energy, Elab, by generating the positron energy, angle
      // distribution in muon rest frame
      y = curand_uniform(&localState);
      A = (-8.0 * y * y + y + 1.) / (y * y - 5. * y - 5.);
      n = (y - 1) * (y * y - 5. * y - 5.);
      r_test = n * (1. - A * cos(omega_a * t)) / 6.;
      r = curand_uniform(&localState);
      if (r >= r_test)
        continue;
      float Elab = Elab_max * y;

      if (fixedElab > 0) {
        Elab = fixedElab;
      }
      // account for acceptance of calorimeter using empirical, energy-dependent
      // calo acceptance

      // TODO Add this parameter and uncomment
      // very simple empirical acceptance, zero below ElabMin, unit above
      // ElabMin
      // float ElabMin = 0.5; // acceptance cutoff
      // if (Elab < ElabMin) continue;

      // simple acceptance paramterization from Aaron see
      // Experiments/g-2/Analysis/Qmethod/functionForTim.C
      if (Elab > 0.7 * Elab_max) {
        r_test = 1.0;
      } else {
        r_test = Elab / (0.7 * Elab_max);
      }
      r = curand_uniform(&localState);
      if (r >= r_test)
        continue;

      // variable ADC is total ADC samples of positron signal at 800 MMz
      // sampling rate with 6.2 GeV max range over 2048 ADC counts
      ADC = GeVToADC * Elab;
      // divide by maximum fraction of positron signal in single 800 MHz bin (is
      // ~0.4 from erfunc plot of 5ns FWHM pulse in peak sample at 800 MHz
      // sampling rate)
      // TODO: Will check this issue back later
      // ADC = ADC/PeakBinFrac;

      // add empirical energy-dependent drift time, see
      // https://muon.npl.washington.edu/elog/g2/Simulation/229 using empirical
      // energy and time relation
      ylab = Elab / Elab_max;
      phase = p0 + p1 * ylab + p2 * ylab * ylab + p3 * ylab * ylab * ylab +
              p4 * ylab * ylab * ylab * ylab +
              p5 * ylab * ylab * ylab * ylab * ylab +
              p6 * ylab * ylab * ylab * ylab * ylab * ylab +
              p7 * ylab * ylab * ylab * ylab * ylab * ylab * ylab +
              p8 * ylab * ylab * ylab * ylab * ylab * ylab * ylab * ylab +
              p9 * ylab * ylab * ylab * ylab * ylab * ylab * ylab * ylab *
                  ylab; // phase in mSR units of omega_a, max
                        // is 5.14662666320800781e+01 msr
      drifttime = anomalousperiod * phase /
                  (2. * Pi * 1000.); // convert the omega_a phase to drift time
                                     // in Q-method histogram bin units
      /*
      tick = tick + drifttime; // add drift time to decay time
      itick = (int)tick;
      //TODO do this later
*/

      // Add To flushHitArray flush buffer
      atomicAdd(&(flushHitArray[iflush * fill_buffer_max_length + itick]), 1.0);
      // Add To energySumArray flush buffer
      atomicAdd(&(energySumArray[itick]),
                Elab); // diagnostic for energy time distributions

      // generate the x, y coordinates of positron hit on calorimeter

      // very simple random (x, y) coordinates
      xcoord = nxseg * curand_uniform(&localState);
      ycoord = nyseg * curand_uniform(&localState);

      // TODO Check this part with Tim
      // rough empirical x-distribution obtained from
      // https://muon.npl.washington.edu/elog/g2/Simulation/258 (Robin) rough
      // empirical y-distribution obtained from
      // https://muon.npl.washington.edu/elog/g2/Simulation/256 (Pete)
      /*
      if ( ylab > 0.7 ) {
              xmax = 185.-533.3*(ylab-0.7);
      } else {
              xmax = 185.;
      }
      xrand = curand_uniform(&localState);
      xcoord = xmax*xrand/25.0; // x-coordinate -> segment/xtal unitsNFlushes
      // put hit information (time, ADC counts, x/y coordinates, hit index) into
      hit array needed for applying pile-up effects
      */
      tickstore[nhit] = tick;
      ADCstore[nhit] = ADC;
      xcoordstore[nhit] = xcoord;
      ycoordstore[nhit] = ycoord;
      iold[nhit] = nhit;

      nhit++; // hit counter
    }

    // we've now got the arrays of time (tick, itick), lab energy / ADC value
    // (Elab / ADC), x/y coordinates (xcoord / ycoord) of positron with
    // empirical acceptance, (x,y) distributions

    // parameters for empirical Gaussian distribution of energy across
    // neighboring segments.

    // https://muon.npl.washington.edu/elog/g2/SLAC+Test+Beam+2016/260 and
    // position where energy in neighboring xtal is 16% (1 sigma) - giving sigma
    // = 0.19 in units of crystal size
    float xsig = SimulationParameters[3], ysig = SimulationParameters[4];
    // test with very small spread
    // float xsig = 0.01, ysig = 0.01;
    // float xsig = 0.5, ysig = 0.5; // test with very large spread
    // float xsig = 0.19, ysig = 0.19; // stanard  distribtion, xtal size units

    // parameters for distributing the ADC counts over time bins of q-method
    // histogram

    float ADCstoresegment[54][NEMAX]; // array used for xtal-by-xtal

    for (int i = 0; i < nhit; i++) { // loop over hits

      tick = tickstore[i]; // in the unit of qBin, qBin width is 75 ns
      ADC = ADCstore[i];
      xcoord = xcoordstore[i];
      ycoord = ycoordstore[i];

      //  add effects of injection flash via time-dependent gain sag
      //  with amplitude of ampFlashGainSag and time constant tauFlashGainSag
      //  to all calo pulses - see
      //  https://gm2-docdb.fnal.gov/cgi-bin/private/RetrieveFile?docid=10818
      //   amplitude of 0.10 is trypical exptrapolation to time t=0 of hot calo
      //   gain sag from beam flash

      float ampFlashGainSag = 0.10, tauFlashGainSag = 1.0e5 / qBinSize;
      flashgainsag = false;
      if (flashgainsag)
        ADC = ADC * (1.0 - ampFlashGainSag * exp(-tick / tauFlashGainSag));

      // need Qmethod bin and time within bin for distributing signal over bins
      // itick is bin of q-method time histogram
      // rtick is time within bin of q-method time histogram
      int itick = (int)tick;      // in the unit of qBin, integer part.
      float rtick = tick - itick; // in the unit of qBin.

      // Time distribution
      float TimeBinFractions[5];
      for (int tempIdx = 0; tempIdx < 5; tempIdx++) {
        TimeBinFractions[tempIdx] = 0.0;
      }
      // TemplateZero in default should be 200, unit is 0.125 ns,
      // so TemplateZero is 25 ns. The template

      // int TickEdgeIndex = int(TemplateZero - rtick * qBinSize * 10);
      // The unit of integrated pulse template is (raw adc bin)/10 = 0.125 ns
      // We use this unit: (raw adc bin)/10 as base unit in the following:
      int TickEdgeIndex = int(TemplateZero - rtick * TIMEDECIMATION * 10);

      // Diagnostics: Try first delta shape pulse
      if (SimulationIntParameters[9] == 1) {
        TimeBinFractions[1] = 1.0;
      }
      // End of Diagnostics.

      // Pulse template based temporal energy distribution.
      // else {
      //   if (TickEdgeIndex > 0) {
      //     TimeBinFractions[0] = pulseTemplate[TickEdgeIndex];
      //   } else {
      //     TimeBinFractions[0] = 0.0;
      //   }

      //   for (int tempIdx = 1; tempIdx < 5; tempIdx++) {
      //     int leftIdx = TickEdgeIndex + (tempIdx - 1) * qBinSize * 10;
      //     int rightIdx = TickEdgeIndex + tempIdx * qBinSize * 10;
      //     if (leftIdx < 0) {
      //       leftIdx = 0;
      //     }
      //     if (leftIdx >= TemplateSize) {
      //       leftIdx = TemplateSize - 1;
      //     }
      //     if (rightIdx >= TemplateSize) {
      //       rightIdx = TemplateSize - 1;
      //     }
      //     TimeBinFractions[tempIdx] =
      //         pulseTemplate[rightIdx] - pulseTemplate[leftIdx];
      //   }
      // }
      else {
        int leftIdx = 160;
        int rightIdx = int(160 + (1 - rtick) * TIMEDECIMATION * 10);
        float FractionSum = 0;
        for (int tempIdx = 0; tempIdx < 3; tempIdx++) {
          TimeBinFractions[tempIdx] =
              pulseTemplate[rightIdx] - pulseTemplate[leftIdx];

          if (TimeBinFractions[tempIdx] > 0) {
            FractionSum = FractionSum + TimeBinFractions[tempIdx];
          }
          leftIdx = rightIdx;
          rightIdx = leftIdx + TIMEDECIMATION * 10;
        }

        for (int tempIdx = 0; tempIdx < 3; tempIdx++) {
          TimeBinFractions[tempIdx] = TimeBinFractions[tempIdx] /
          FractionSum;
        }
      }

      // End of temporal distribution.

      // we're now going to distribute the ADC signal accross x,y segments and
      // time bins filling the array fillSumArray of size time bins * xsegs *
      // ysegs

      // loop over the array of xtals and distribute the total ADC counts (ADC)
      // to each xtal (ADC segment) using the hit coordinates xcoord, ycoords
      // and spreads xsig, ysig.
      // Calculating the gaussian integral for each x or y segments
      float fsegmentsum =
          0.0; // diagnostic parameter for distribution of energy over segments
      float SegmentX[NXSEG];
      float SegmentY[NYSEG];
      for (int ix = 0; ix < nxseg; ix++) {
        SegmentX[ix] = 0.5 * (-erfcf((ix + 1.0 - xcoord) / (sqrt(2.) * xsig)) +
                              erfcf((ix - xcoord) / (sqrt(2.) * xsig)));
      }
      for (int iy = 0; iy < nyseg; iy++) {
        SegmentY[iy] = 0.5 * (-erfcf((iy + 1.0 - ycoord) / (sqrt(2.) * ysig)) +
                              erfcf((iy - ycoord) / (sqrt(2.) * ysig)));
      }
      for (int ix = 0; ix < nxseg; ix++) {
        for (int iy = 0; iy < nyseg;
             iy++) { // calc energy in segment (assume aGaussian distribution
                     // about xc, yc)
          float fsegment = SegmentX[ix] * SegmentY[iy];
          float ADCsegment = fsegment * ADC;
          fsegmentsum += fsegment;

          // array offset needed for storing xtal hits in samples array
          int xysegmentoffset = (ix + iy * nxseg) * fill_buffer_max_length;

          // Time distribution
          for (int k = 0; k < 5; k++) {
            // for (int k = 0; k < 3; k++) {
            int kk = k + itick - 1;
            if (kk < 0 || kk >= fill_buffer_max_length)
              continue;
            float ADCfrac = ADCsegment * TimeBinFractions[k];
            //	     if (ADCfrac<-100.0)printf("%f
            //%f\n",fsegment,TimeBinFractions[k]);

            // printf("xtal %i, ADCfrac %f\n", ix+iy*nxseg, ADCfrac);
            atomicAdd(
                &(flushQTruthArray[iflush * nsegs * fill_buffer_max_length +
                                   xysegmentoffset + kk]),
                ADCfrac /
                    TIMEDECIMATION); // divide by time decimation to account
                                     // for data processing calculates the
                                     // time-decimated average not sum
            atomicAdd(
                &(flushQArray[iflush * nsegs * fill_buffer_max_length +
                              xysegmentoffset + kk]),
                ADCfrac /
                    TIMEDECIMATION); // Keep this array only to this fill and
                                     // use it for analysis, add pedestal later
          }
          // OBSOLETE do time smearing of positron pulse over several contiguous
          // time bins, just loop over bins k-1, k, k+1 as negligible in other
          // bins

        } // end of y-distribution loop
      }   // end of x-distribution loop

    } // end of time-ordered hits hits

    // Add pedestal
    // for (int ix = 0; ix < nxseg; ix++) {
    //   for (int iy = 0; iy < nyseg;
    //        iy++) { // calc energy in segment (assume aGaussian distribution
    //                // about xc, yc)
    //     // array offset needed for storing xtal hits in samples array
    //     int xysegmentoffset = (ix + iy * nxseg) * fill_buffer_max_length;

    //     for (int k = 35; k < fill_buffer_max_length; k++) {
    //       atomicAdd(&(flushQArray[iflush * nsegs * fill_buffer_max_length +
    //                               xysegmentoffset + k]),
    //                 pedestalTemplate[k]);
    //     }
    //     //	 printf("\n ");

    //   } // end of y-distribution loop
    // }   // end of x-distribution loop

    // state index for random number generator
    state[idx] = localState;

  } // end of if idx < nFillsPerFlush
}

namespace QSimulation {
QSim::QSim(const std::map<std::string, int> &tIntParameters,
           const std::map<std::string, float> &tFloatParameters,
           const std::map<std::string, std::string> &tStringParameters,
           long long int tSeed) {
  // Import Parameters
  IntParameters = tIntParameters;
  FloatParameters = tFloatParameters;
  StringParameters = tStringParameters;

  // Additional Parameters
  IntParameters["FillBufferMaxLength"] = nsPerFill / qBinSize;
  int deviceIdx = 0;
  // Template
  if (StringParameters["Pulse Shape"].compare("Delta") == 0) {
    IntegratedPulseTemplate = std::vector<float>(IntParameters["TemplateSize"]);
    for (int i = IntParameters["TemplateZero"];
         i < IntParameters["TemplateSize"]; i++) {
      IntegratedPulseTemplate[i] = 1.0;
    }
  } else if (StringParameters["Pulse Shape"].compare("Template") == 0) {
    int ret = IntegratePulseTemplate(StringParameters["Pulse Template Path"],
                                     25, IntParameters["TemplateSize"],
                                     IntParameters["TemplateZero"]);
  } else {
    IntegratedPulseTemplate = std::vector<float>(IntParameters["TemplateSize"]);
    for (int i = IntParameters["TemplateZero"];
         i < IntParameters["TemplateSize"]; i++) {
      IntegratedPulseTemplate[i] = 1.0;
    }
  }

  if (StringParameters["Pedestal Shape"].compare("Flat") == 0) {
    PedestalTemplate =
        std::vector<float>(IntParameters["FillBufferMaxLength"], 0.0);
  } else if (StringParameters["Pulse Shape"].compare("Template") == 0) {
    PedestalTemplate =
        std::vector<float>(IntParameters["FillBufferMaxLength"], 0.0);
    int ret =
        LoadPedestalTemplate(StringParameters["Pedestal Template Path"], 0);
  } else {
    PedestalTemplate =
        std::vector<float>(IntParameters["FillBufferMaxLength"], 0.0);
  }

  // Allocate Derive memory for templates
  cudaMalloc((void **)&d_IntegratedPulseTemplate,
             IntParameters["TemplateSize"] * sizeof(float));
  cudaMalloc((void **)&d_PedestalTemplate,
             IntParameters["FillBufferMaxLength"] * sizeof(float));

  cudaMemcpy(d_IntegratedPulseTemplate, &IntegratedPulseTemplate[0],
             IntParameters["TemplateSize"] * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_PedestalTemplate, &PedestalTemplate[0],
             IntParameters["FillBufferMaxLength"] * sizeof(float),
             cudaMemcpyHostToDevice);

  // Init Parameter Arrays
  InitParameters();
  // cudaSetDevice(deviceIdx);
  // Allocate Derive memory for parameters

  cudaMalloc((void **)&d_SimulationParameters,
             SimulationParameters.size() * sizeof(float));
  cudaMalloc((void **)&d_SimulationIntParameters,
             SimulationIntParameters.size() * sizeof(int));

  cudaMemcpy(d_SimulationParameters, &SimulationParameters[0],
             SimulationParameters.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_SimulationIntParameters, &SimulationIntParameters[0],
             SimulationIntParameters.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  // Arrays

  int nFillsPerBatch =
      IntParameters["NFlushesPerBatch"] * IntParameters["NFillsPerFlush"];
  int nFlushesPerBatch = IntParameters["NFlushesPerBatch"];

  ArraySizes["pulseTemplate"] =
      IntParameters["TemplateSize"] * sizeof(float); // pulse template array
  ArraySizes["pedestalTemplate"] = IntParameters["FillBufferMaxLength"] *
                                   sizeof(float); // pulse template array

  ArraySizes["EnergySumArray"] =
      IntParameters["FillBufferMaxLength"] * sizeof(float);
  ArraySizes["FlushHitArray"] =
      nFlushesPerBatch * IntParameters["FillBufferMaxLength"] * sizeof(float);
  ArraySizes["FlushQTruthArray"] = nFlushesPerBatch * nsegs *
                                   IntParameters["FillBufferMaxLength"] *
                                   sizeof(float);
  ArraySizes["FlushQArray"] = nFlushesPerBatch * nsegs *
                              IntParameters["FillBufferMaxLength"] *
                              sizeof(float);

  // get some cuda device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceIdx);
  printf("Device Number: %d\n", deviceIdx);
  printf("Device name: %s\n", prop.name);
  printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
  printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("\n\n");

  // Allocate memories
  cudaMalloc((void **)&d_state, nsegs * nFillsPerBatch * sizeof(curandState));

  for (auto it = ArraySizes.begin(); it != ArraySizes.end(); ++it) {
    auto Name = it->first;
    auto Size = it->second;
    HostArrays[Name] = (float *)malloc(Size);
    cudaMalloc((void **)&DeviceArrays[Name], Size);
  }

  cudaError err;

  int nblocks = nFillsPerBatch / IntParameters["NThreadsPerBlock"] + 1;
  long long int rand_seed = tSeed;
  if (rand_seed == 0) {
    rand_seed = time(NULL);
  }
  init_rand<<<nblocks, IntParameters["NThreadsPerBlock"]>>>(d_state, 0,
                                                            rand_seed);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Cuda failure with user kernel function init_rand  %s:%d: '%s'\n",
           __FILE__, __LINE__, cudaGetErrorString(err));
    exit(0);
  }
}

QSim::~QSim() {
  for (auto it = HostArrays.begin(); it != HostArrays.end(); ++it) {
    free(it->second);
  }
  for (auto it = DeviceArrays.begin(); it != DeviceArrays.end(); ++it) {
    cudaFree(it->second);
  }

  cudaFree(d_SimulationParameters);
  cudaFree(d_SimulationIntParameters);

  for (auto &&it : AnaModules) {
    delete it.second;
  }
}

int QSim::Simulate(int NFlushes) {
  cudaError err;
  int NSim = NFlushes / IntParameters["NFlushesPerBatch"] + 1;
  // Clean device memory
  for (auto it = ArraySizes.begin(); it != ArraySizes.end(); ++it) {
    auto Name = it->first;
    auto Size = it->second;
    cudaMemset(DeviceArrays[Name], 0.0, Size);
  }
  // Call Alanalysis functions
  for (auto &&it : AnaModules) {
    (it.second)->DeviceMemoryReset();
  }

  int nFillsPerBatch =
      IntParameters["NFlushesPerBatch"] * IntParameters["NFillsPerFlush"];
  for (int i = 0; i < NSim; i++) {
    // Simulate
    // make the fills
    int nblocks = nFillsPerBatch / IntParameters["NThreadsPerBlock"] + 1;

    // std::cout << nblocks<<std::endl;
    std::cout << "Simulating " << i * IntParameters["NFlushesPerBatch"]
              << " flushes " << std::endl;

    if (IntParameters["AccumulateMode"] == 0) {
      cudaMemset(DeviceArrays["FlushHitArray"], 0.0,
                 ArraySizes["FlushHitArray"]);
      cudaMemset(DeviceArrays["FlushQTruthArray"], 0.0,
                 ArraySizes["FlushQTruthArray"]);
      cudaMemset(DeviceArrays["FlushQArray"], 0.0, ArraySizes["FlushQArray"]);
    }

    make_randfill<<<nblocks, IntParameters["NThreadsPerBlock"]>>>(
        d_state, d_IntegratedPulseTemplate, d_PedestalTemplate,
        DeviceArrays["EnergySumArray"], DeviceArrays["FlushHitArray"],
        DeviceArrays["FlushQTruthArray"], DeviceArrays["FlushQArray"],
        d_SimulationIntParameters, d_SimulationParameters);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf(
          "Cuda failure with user kernel function make)randfill %s:%d: '%s'\n",
          __FILE__, __LINE__, cudaGetErrorString(err));
      exit(0);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf(
          "Cuda failure with user kernel function make)randfill %s:%d: '%s'\n",
          __FILE__, __LINE__, cudaGetErrorString(err));
      exit(0);
    }

    // Call Alanalysis functions
    for (auto &it : AnaModules) {
      it.second->FlushAnalysis(&HostArrays, &DeviceArrays, &ArraySizes);
    }
  }

  // Copy back to host memory
  int n = 0;
  for (auto it = ArraySizes.begin(); it != ArraySizes.end(); ++it) {
    auto Name = it->first;
    auto Size = it->second;
    cudaMemcpy(HostArrays[Name], DeviceArrays[Name], Size,
               cudaMemcpyDeviceToHost);
    //      std::cout<< n << " "<<Name<<" "<<Size<<std::endl;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf(
          "Cuda failure with user kernel function make)randfill %s:%d: '%s'\n",
          __FILE__, __LINE__, cudaGetErrorString(err));
      exit(0);
    }
    n++;
  }

  // Call Alanalysis functions
  for (auto &it : AnaModules) {
    it.second->EndAnalysis(&HostArrays, &ArraySizes);
  }

  return 0;
}

int QSim::Output(int RunNumber) {
  std::vector<double> LastFlushQHist;
  std::vector<double> LastFlushQTruthHist;
  this->GetCaloArray("FlushQArray", LastFlushQHist);
  this->GetCaloArray("FlushQTruthArray", LastFlushQTruthHist, false);

  unsigned int N = LastFlushQHist.size();

  TH1 *hLastFlushQ =
      new TH1D("LastFlushQHist", "LastFlushQHist", N, 0, N * 0.075);
  for (unsigned int i = 0; i < N; i++) {
    if (i > 3100)
      break;
    hLastFlushQ->SetBinContent(i, LastFlushQHist[i]);
  }

  TH1 *hLastFlushQTruth =
      new TH1D("LastFlushQTruthHist", "LastFlushQTruthHist", N, 0, N * 0.075);
  for (unsigned int i = 0; i < N; i++) {
    if (i > 3100)
      break;
    hLastFlushQTruth->SetBinContent(i, LastFlushQTruthHist[i]);
    // h->SetBinError(i,sqrt(QHist[i]));
  }

  auto PulseTemplate = this->GetIntegratedPulseTemplate();
  TH1 *hTemplate = new TH1D("Template", "Template", 2000, -20, 180);
  for (int i = 0; i < 2000; i++) {
    hTemplate->SetBinContent(i, PulseTemplate[i]);
  }

  auto PedestalTemplate = this->GetPedestalTemplate();
  TH1 *hPedTemplate =
      new TH1D("PedTemplate", "PedTemplate", PedestalTemplate.size(), 0,
               PedestalTemplate.size());
  for (int i = 0; i < PedestalTemplate.size(); i++) {
    hPedTemplate->SetBinContent(i, PedestalTemplate[i]);
  }

  TFile *FileOut =
      new TFile(Form("SimulatorOut_%04d.root", RunNumber), "recreate");
  hLastFlushQ->Write();
  hLastFlushQTruth->Write();
  hTemplate->Write();
  hPedTemplate->Write();
  FileOut->Close();

  delete FileOut;
  delete hLastFlushQ;
  delete hLastFlushQTruth;
  delete hTemplate;
  delete hPedTemplate;

  // Call Analyzer output

  for (auto &it : AnaModules) {
    it.second->Output(RunNumber);
  }

  return 0;
}

int QSim::GetArray(std::string ArrayName, std::vector<double> &Output) {
  auto Size = ArraySizes[ArrayName];
  Output.resize(Size);
  auto ptr = HostArrays[ArrayName];
  for (int i = 0; i < Size; i++) {
    Output[i] = ptr[i];
  }
  return 0;
}

int QSim::GetCaloArray(std::string ArrayName, std::vector<double> &Output,
                       bool BatchSum) {
  Output.clear();
  Output.resize(IntParameters["FillBufferMaxLength"], 0.0);
  auto ptr = HostArrays[ArrayName];

  int nFlushesPerBatch = IntParameters["NFlushesPerBatch"];

  for (unsigned int k = 0; k < nFlushesPerBatch; k++) {
    for (unsigned int j = 0; j < nsegs; j++) {
      for (unsigned int i = 0; i < IntParameters["FillBufferMaxLength"]; i++) {
        Output[i] +=
            ptr[(k * nsegs + j) * IntParameters["FillBufferMaxLength"] + i];
      }
    }
    if (!BatchSum) {
      break;
    }
  }

  return 0;
}

// Private Functions
int QSim::InitParameters() {
  SimulationParameters.resize(5);
  SimulationIntParameters.resize(10);

  SimulationParameters[0] = FloatParameters["Omega_a"];
  SimulationParameters[1] = FloatParameters["Noise"];
  SimulationParameters[2] = FloatParameters["Elab"];
  SimulationParameters[3] = FloatParameters["Sigma_X"];
  SimulationParameters[4] = FloatParameters["Sigma_Y"];
  // std::cout << "SimulationParameter[2]: " << SimulationParameters[2]
  //           << std::endl;
  SimulationIntParameters[0] = IntParameters["NThreadsPerBlock"];
  SimulationIntParameters[1] = IntParameters["FillBufferMaxLength"];
  SimulationIntParameters[2] = IntParameters["NElectronsPerFill"];
  SimulationIntParameters[3] = IntParameters["NFillsPerFlush"];
  SimulationIntParameters[4] = IntParameters["NFlushesPerBatch"];
  SimulationIntParameters[5] = IntParameters["TemplateSize"];
  SimulationIntParameters[6] = IntParameters["TemplateZero"];
  SimulationIntParameters[7] = IntParameters["FillNoiseSwitch"];
  SimulationIntParameters[8] = IntParameters["FlashGainSagSwitch"];
  SimulationIntParameters[9] = IntParameters["TimeDiagnostics"];

  std::cout << "Int Parameters:\n";
  for (auto pair : IntParameters) {
    std::cout << pair.first << ": " << pair.second << "\n";
  }
  std::cout << "\n";
  std::cout << "Float Parameters:\n";
  for (auto pair : FloatParameters) {
    std::cout << pair.first << ": " << pair.second << "\n";
  }
  return 0;
}

int QSim::IntegratePulseTemplate(std::string TemplatePath, int CrystalId,
                                 int TemplateSize, int TemplateZero) {
  std::string FileName =
      TemplatePath + "/template" + std::to_string(CrystalId) + ".root";
  TFile *TemplateFile = new TFile(FileName.c_str(), "read");
  auto TemplateSpline = (TSpline3 *)TemplateFile->Get("masterSpline");

  IntegratedPulseTemplate.clear();
  IntegratedPulseTemplate.resize(TemplateSize);

  float AccumulatedVal = 0.0;
  for (int i = 0; i < TemplateSize; i++) {
    float t = static_cast<float>(i - TemplateZero) / 10.0;
    float val = TemplateSpline->Eval(t);
    AccumulatedVal += val;
    IntegratedPulseTemplate[i] = AccumulatedVal;
  }
  float norm = IntegratedPulseTemplate[TemplateSize - 1];
  for (int i = 0; i < TemplateSize; i++) {
    IntegratedPulseTemplate[i] /= norm;
  }

  TemplateFile->Close();
  delete TemplateFile;

  return 0;
}

int QSim::LoadPedestalTemplate(std::string TemplatePath, int CrystalId) {
  std::string FileName = TemplatePath + "/PedTemplate" + ".root";
  TFile *TemplateFile = new TFile(FileName.c_str(), "read");
  auto PedTemplate = (TH1 *)TemplateFile->Get("hCalo_Pedestal_10");

  unsigned int Size = PedTemplate->GetNbinsX();

  if (Size > PedestalTemplate.size()) {
    Size = PedestalTemplate.size();
  }

  double norm = 2e-4;

  for (int i = 0; i < Size; i++) {
    PedestalTemplate[i] = PedTemplate->GetBinContent(i) * norm;
  }
  TemplateFile->Close();
  delete TemplateFile;

  return 0;
}

} // namespace QSimulation
