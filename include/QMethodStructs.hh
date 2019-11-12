#ifndef QMETHODSTRUCTS_HH
#define QMETHODSTRUCTS_HH

#define NCALO 24
#define NCRYSTAL 54
#define NQBINSOUT 3180 //Has to be even so that the memory alignment issue is not there.
#define OFFSET 20000

namespace gm2analyses
{

struct QHistCrystalEvent_t
{
  double Pedestal[NQBINSOUT]; 
  double Signal[NQBINSOUT]; 
  double SignalErr[NQBINSOUT]; 
  double CrystalHitEnergy[NQBINSOUT]; //Energy determined through crystalHit product
  double Raw[NQBINSOUT]; 
  int BinTag[NQBINSOUT]; // tag =  number of hits for each bin
};

struct QHistCaloEvent_t
{
  double Signal[NQBINSOUT];
  double SignalErr[NQBINSOUT]; 
  double CrystalHitEnergy[NQBINSOUT]; //Energy determined through crystalHit product
  int BinTag[NQBINSOUT];// for each calo, tag = number of hits summing over all crystals for each bin
};

struct QHistCrystalSum_t
{
  double Signal[NQBINSOUT]; 
  double SignalErr[NQBINSOUT]; 
  double CrystalHitEnergy[NQBINSOUT]; //Energy determined through crystalHit product
};

struct QHistCaloSum_t
{
  double Signal[NQBINSOUT]; 
  double SignalErr[NQBINSOUT]; 
  double CrystalHitEnergy[NQBINSOUT]; //Energy determined through crystalHit product
};

struct QMethodInfo_t
{
  double EnergyThreshold;
  int PedestalMode; //Ran : 0 , Tim : 1
  int ThresholdMode; //Whole Calo: 0, Crystal: 1
};

}
#endif
