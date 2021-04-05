#include "AnalysisModuleBase.h"

#include <map>
#include <string>
#include <vector>

namespace QAnalysis {
class TruthAnalysisModule : public AnalysisModule {
public:
  TruthAnalysisModule(std::string Name,
                   const std::map<std::string, int> &tIntParameters,
                   const std::map<std::string, float> &tFloatParameters,
                   const std::map<std::string, std::string> &tStringParameters,
                   int nFlushesPerBatch, int FillMaxLength);

  int FlushAnalysis(std::map<std::string, float *> *SimulatorHostArrays,
                    std::map<std::string, float *> *SimulatorDeviceArrays,
                    std::map<std::string, int> *SimulatorArraySizes) override;
  int EndAnalysis(std::map<std::string, float *> *SimulatorHostArrays,
                  std::map<std::string, int> *SimulatorArraySizes) override;
  int Output(int RunNumber) override;

  int GetArray(std::string ArrayName, std::vector<double> &Output);
  int GetCaloArray(std::string ArrayName, std::vector<double> &Output,
                   bool BatchSum = true);

protected:
  int InitParameters();
};
} // end namespace QAnalysis

