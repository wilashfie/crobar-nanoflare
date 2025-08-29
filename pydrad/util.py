import xml.etree.ElementTree as ET
import numpy as np
import json

def convert_value(text):
    """Convert text to appropriate data type"""
    if text is None:
        return None
    text = text.strip()
    if text.lower() in ['true', 'false']:
        return text.lower() == 'true'
    try:
        if '.' in text and 'e' not in text.lower():
            return float(text.split()[0])  # Handle values like "86400.0 s"
        elif 'e' in text.lower():
            return float(text.split()[0])  # Handle scientific notation
        else:
            return int(text.split()[0])
    except (ValueError, IndexError):
        return text  # Return as string if conversion fails

def parse_xml_config(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    
    config_data = {}
    
    for child in root:
        if child.tag == 'dem':
            # Handle nested dem structure
            dem_data = {}
            for dem_child in child:
                if dem_child.tag == 'use_new_method':
                    dem_data['use_new_method'] = convert_value(dem_child.text)
                elif dem_child.tag == 'temperature':
                    dem_data['temperature'] = dem_child.attrib
            config_data['dem'] = dem_data
            
        elif child.tag == 'heating':
            # Handle nested heating structure
            heating_data = {}
            for heating_child in child:
                if heating_child.tag == 'background':
                    heating_data['background'] = convert_value(heating_child.text)
                elif heating_child.tag == 'partition':
                    heating_data['partition'] = convert_value(heating_child.text)
                elif heating_child.tag == 'events':
                    events = []
                    for event in heating_child:
                        events.append(event.attrib)
                    heating_data['events'] = events
            config_data['heating'] = heating_data
            
        else:
            # Handle simple elements
            config_data[child.tag] = convert_value(child.text)
    
    return config_data


def add_unit(elem):
    if isinstance(elem,str):
        if elem.strip().lower() in ["true","false"]:
            return elem.strip().lower()=="true"
        try:
            value, unit = elem.split()[0], "".join(elem.split()[1:])
            return float(value)*u.Unit(unit)
        except:
            pass
        try:
            return float(elem.strip())
        except:
            return elem.strip()
    elif isinstance(elem,dict):
        for key in elem:
            return add_unit(elem[key])
    else:
        return None

        
def get_events(heating,loop_length,loc_fraction=0.5,scale_height_fraction=0.25):
    '''
    heating: heating event dictionary from parse_xml_config(.xml) output 
    loop_length: in Mm
    location: fraction along loop (0.5 = center)
    scale_height: Gaussian heating function width -- fraction of loop 
    '''

    location = loop_length * loc_fraction
    scale_height = loop_length * scale_height_fraction
    heating_scale = loop_length/scale_height/np.sqrt(2*np.pi)

    events=[]
    for event in heating['events']:
        
        time_start = add_unit(event['rise_start'])
        rise_duration = add_unit(event['rise_end'])-add_unit(event['rise_start'])
        decay_duration = add_unit(event['decay_end'])-add_unit(event['decay_start'])
        total_duration= rise_duration+decay_duration

        mag = add_unit(event['magnitude'])
        if scale_height < (1e100*u.Mm):
            rate = mag * heating_scale
        else:
            rate = mag

        events.append({'time_start' : time_start,
                      'rise_duration' : rise_duration,
                        'decay_duration' : decay_duration,
                       'total_duration': total_duration,
                       'location' : location,
                       'scale_height' : scale_height,
                       'rate' : rate,
                      })
    return events  




# Usage
#config_data = parse_xml_config(loop_config)

'''
print(f"Total time: {config_data['total_time']}")
print(f"Force single fluid: {config_data['force_single_fluid']}")
print(f"DEM use new method: {config_data['dem']['use_new_method']}")
print(f"Temperature bins: {config_data['dem']['temperature']['bins']}")
print(f"Heating background: {config_data['heating']['background']}")
print(f"Number of events: {len(config_data['heating']['events'])}")
'''


import json
from astropy import units as u

def convert_quantities_to_serializable(data):
    """Recursively convert astropy quantities to serializable format"""
    if isinstance(data, dict):
        return {key: convert_quantities_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_quantities_to_serializable(item) for item in data]
    elif hasattr(data, 'unit') and hasattr(data, 'value'):  # It's a Quantity
        return {'value': float(data.value), 'unit': str(data.unit)}
    else:
        return data  # Regular data (strings, numbers, booleans)

def convert_serializable_to_quantities(data):
    """Recursively convert serializable format back to quantities"""
    if isinstance(data, dict):
        if 'value' in data and 'unit' in data and len(data) == 2:
            # This looks like a serialized quantity
            return data['value'] * u.Unit(data['unit'])
        else:
            # Regular dictionary - recurse into it
            return {key: convert_serializable_to_quantities(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_serializable_to_quantities(item) for item in data]
    else:
        return data  # Regular data