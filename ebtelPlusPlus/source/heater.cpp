/*
heater.cpp
Class defnition for heater object
*/

#include "heater.h"

Heater::Heater(tinyxml2::XMLElement * heating_node)
{
  //Set basic parameters
  background = std::stod(get_element_text(heating_node,"background"));
  partition = std::stod(get_element_text(heating_node,"partition"));

  //Set heating parameters
  tinyxml2::XMLElement * events = get_element(heating_node,"events");
  for(tinyxml2::XMLElement *child = events->FirstChildElement();child != NULL;child=child->NextSiblingElement())
  {
    time_start_rise.push_back(std::stod(child->Attribute("rise_start")));
    time_end_rise.push_back(std::stod(child->Attribute("rise_end")));
    time_start_decay.push_back(std::stod(child->Attribute("decay_start")));
    time_end_decay.push_back(std::stod(child->Attribute("decay_end")));
    magnitude.push_back(std::stod(child->Attribute("magnitude")));
  }
  num_events = magnitude.size();
}

Heater::Heater(void)
{
  // Default constructor
}

Heater::~Heater(void)
{
  //Destructor--free some stuff here
}

double Heater::Get_Heating(double time)
{
  double heat = background;
  for(int i=0;i<num_events;i++)
  {
    if(time >= time_start_rise[i] && time < time_end_rise[i])
    {
      heat += magnitude[i]*(time - time_start_rise[i])/(time_end_rise[i] - time_start_rise[i]);
    }
    else if(time >= time_end_rise[i] && time < time_start_decay[i])
    {
      heat += magnitude[i];
    }
    else if(time >= time_start_decay[i] && time < time_end_decay[i])
    {
      heat += magnitude[i]*(time_end_decay[i] - time)/(time_end_decay[i] - time_start_decay[i]);
    }
  }

  return heat;
}

double Heater::Get_Time_To_Next_Heating_Change(double time)
{
  double tau = std::numeric_limits<double>::max();
  for(int i=0;i<num_events;i++)
  {
    if(time < time_start_rise[i])
    {
      tau = std::fmin( tau, time_start_rise[i] - time);
    }
    else if(time < time_end_rise[i])
    {
      tau = std::fmin( tau, time_end_rise[i] - time);
    }
    else if(time < time_start_decay[i])
    {
      tau = std::fmin( tau, time_start_decay[i] - time);
    }
    else if(time < time_end_decay[i])
    {
      tau = std::fmin( tau, time_end_decay[i] - time);
    }
  }

  return tau;
}
