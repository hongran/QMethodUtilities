#ifndef UTIL_H
#define UTIL_H

#include <string>
#include "json.hpp"

using json = nlohmann::json;

int ImportJson(std::string inpath,json &jObj){
  std::ifstream infile;
  infile.open(inpath);
  if( infile.fail() ){
    std::cout << "[gm2fieldUtil::Import::ImportJSON]: Cannot open the file: " << inpath << std::endl;
    return -1;
  }else{
    infile >> jObj;
  }
  return 0;
}


template<typename T> T GetValueFromJson(const json& InputStruct,std::string key)
{
  if (InputStruct.find(key.c_str())!=InputStruct.end())
  {
    T result = InputStruct[key.c_str()];
    return result;
  }else{
    std::cout <<"Warning: key "<<key<<" is not found. Set to 0."<<std::endl;
  }
  return 0;
}

json GetStructFromJson(const json& InputStruct,std::string key)
{
  if (InputStruct.find(key.c_str())!=InputStruct.end())
  {
    json result = InputStruct[key.c_str()];
    return result;
  }else{
    std::cout <<"Warning: key "<<key<<" is not found. Set to empty json struct"<<std::endl;
  }
  json empty;
  return empty;
}
#endif
