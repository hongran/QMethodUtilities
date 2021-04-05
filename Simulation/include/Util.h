#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <iostream>
#include <fstream>
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

void JsonToStructs(const json & Config, std::map<std::string,int>& IntParameters, std::map<std::string,float>& FloatParameters, std::map<std::string, std::string>& StringParameters)
{
  json ConfigInt = Config["Integer Parameters"];
  json ConfigFloat = Config["Float Parameters"];
  json ConfigString = Config["String Parameters"];

  for (json::iterator it = ConfigInt.begin(); it != ConfigInt.end(); ++it)
  {
    IntParameters[it.key()] = GetValueFromJson<int>(ConfigInt,it.key());
  }

  for (json::iterator it = ConfigFloat.begin(); it != ConfigFloat.end(); ++it)
  {
    FloatParameters[it.key()] = GetValueFromJson<float>(ConfigFloat,it.key());
  }

  for (json::iterator it = ConfigString.begin(); it != ConfigString.end(); ++it)
  {
    StringParameters[it.key()] = GetValueFromJson<std::string>(ConfigString,it.key());
  }
}


#endif
