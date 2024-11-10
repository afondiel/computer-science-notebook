# Satellite - Notes

## Table of Contents
- Introduction
- Key Concepts
- Applications
- Architecture Pipeline
- Frameworks / Key Theories or Models
- How Satellites Work
- Types of Satellites & Variations
- Self-Practice / Hands-On Examples
- Pitfalls & Challenges
- Feedback & Evaluation
- Tools, Libraries & Frameworks
- Hello World! (Practical Example)
- Advanced Exploration
- Zero to Hero Lab Projects
- Continuous Learning Strategy
- References

## Introduction
Satellites are artificial objects launched into orbit around Earth or other celestial bodies to collect data, communicate, or monitor various aspects of their surroundings.

### Key Concepts
- **Orbit**: The path that a satellite follows around a planet, typically defined as Low Earth Orbit (LEO), Medium Earth Orbit (MEO), or Geostationary Earth Orbit (GEO).
- **Telemetry**: Data collected by satellites for monitoring and control, often sent back to ground stations.
- **Downlink/Uplink**: Communication paths where downlink refers to data sent from the satellite to Earth, and uplink is data sent from Earth to the satellite.
- **Payload**: Equipment or instruments on board the satellite that gather specific data, such as cameras or sensors.
- **Misconceptions**: A common misconception is that all satellites are in GEO; however, most Earth observation satellites orbit in LEO or MEO for better imaging resolution and reduced signal delay.

### Applications
1. **Earth Observation**: Monitoring environmental changes, agriculture, disaster management.
2. **Telecommunications**: Providing internet, TV, and radio signals globally.
3. **Navigation**: Systems like GPS, GLONASS, and Galileo provide global positioning data.
4. **Military Surveillance**: Monitoring strategic areas for defense purposes.
5. **Space Exploration**: Studying other planets, moons, and the solar system.

## Architecture Pipeline
```mermaid
graph LR
    A[Mission Design] --> B[Satellite Development]
    B --> C[Launch and Deployment]
    C --> D[Operational Phase]
    D --> E[Decommissioning and Deorbiting]
```

### Description
1. **Mission Design**: Defining objectives, selecting payloads, and determining the orbit.
2. **Satellite Development**: Designing and building the satellite, including payload integration and testing.
3. **Launch and Deployment**: The satellite is launched into its designated orbit, often by a rocket.
4. **Operational Phase**: Collecting and transmitting data, while ground stations control satellite functions.
5. **Decommissioning and Deorbiting**: When a satellite’s mission is complete, it is either deorbited or placed in a graveyard orbit.

## Frameworks / Key Theories or Models
1. **Kepler’s Laws of Planetary Motion**: Govern satellite orbits and predict satellite position and speed.
2. **Orbital Mechanics**: Study of forces and motion that keep satellites in orbit, crucial for understanding satellite placement and trajectory adjustments.
3. **Attitude Control Systems (ACS)**: Technologies that maintain the satellite's orientation using gyroscopes, reaction wheels, and thrusters.
4. **Communication Protocols**: Standards like S-band, X-band, and Ku-band are used for satellite data transmission.
5. **Ground Segment Systems**: Infrastructure on Earth for monitoring and communicating with satellites.

## How Satellites Work
1. **Launch and Orbit Insertion**: Rockets deliver satellites to their designated orbit.
2. **Stabilization and Orientation**: The satellite stabilizes itself using ACS, orienting solar panels and instruments.
3. **Data Collection**: Payload sensors and instruments gather data or images.
4. **Data Transmission**: Collected data is downlinked to ground stations, while commands are uplinked for satellite control.
5. **Position and Trajectory Adjustments**: Periodic thruster activations adjust the satellite's orbit if needed.

## Types of Satellites & Variations
- **Communications Satellites**: Transmit internet, radio, and TV signals; usually in GEO.
- **Earth Observation Satellites**: Monitor the Earth’s surface, climate, and atmosphere.
- **Navigation Satellites**: Provide location services; operate in MEO (e.g., GPS).
- **Weather Satellites**: Track and forecast weather patterns.
- **Science/Exploration Satellites**: Observe other celestial bodies or space phenomena.

## Self-Practice / Hands-On Examples
1. **Orbit Prediction Exercises**: Use Kepler’s laws and basic orbital mechanics to predict satellite paths.
2. **Satellite Imagery Analysis**: Work with open-source datasets (e.g., Landsat) to analyze Earth imagery.
3. **Simple Ground Station Simulation**: Experiment with antenna design to understand signal tracking.
4. **Space Mission Planning**: Design a mission plan for a satellite, including payload selection and orbit type.
5. **Telemetry Data Analysis**: Practice decoding sample telemetry data from public sources.

## Pitfalls & Challenges
- **Signal Interference**: Physical obstacles and atmospheric conditions can disrupt satellite signals.
- **Space Debris**: Potential collisions with debris, which can damage or destroy satellites.
- **Power Limitations**: Satellites rely on solar power; issues can arise when in the Earth's shadow.
- **Orbital Decay**: Satellites in LEO experience gradual orbit decay, requiring periodic adjustments.
- **Cost**: Launching and maintaining satellites is costly, especially in high orbits.

## Feedback & Evaluation
- **Telemetry Data Monitoring**: Analyze and troubleshoot telemetry data for satellite health checks.
- **Image Resolution Evaluation**: Assess image quality from satellite data to understand payload capabilities.
- **Orbital Simulation**: Use simulators to visualize satellite paths and gain insight into orbital dynamics.

## Tools, Libraries & Frameworks
1. **GMAT (General Mission Analysis Tool)**: Open-source software for planning satellite missions.
2. **STK (Systems Tool Kit)**: Software for satellite and space mission analysis.
3. **OpenStreetMap**: Provides ground-based data for integration with satellite imagery.
4. **SPICE Toolkit**: NASA’s toolkit for planning satellite missions and tracking trajectories.
5. **PyEphem and Skyfield**: Python libraries for calculating satellite positions and tracking orbits.

## Hello World! (Practical Example)
```python
from skyfield.api import Topos, load

# Load satellite data
satellite_url = 'https://celestrak.com/NORAD/elements/stations.txt'
satellites = load.tle_file(satellite_url)
by_name = {sat.name: sat for sat in satellites}
satellite = by_name['ISS (ZARYA)']

# Set observer location (lat/long)
observer = Topos('47.6062 N', '122.3321 W')
ts = load.timescale()
time = ts.now()

# Calculate the position of the satellite from observer's location
satellite_at = satellite.at(time)
position = satellite_at.subpoint()
print('Latitude:', position.latitude.degrees)
print('Longitude:', position.longitude.degrees)
```

## Advanced Exploration
1. **Read**: "Orbital Mechanics for Engineering Students" by Howard Curtis.
2. **Watch**: Space-related courses on edX, Coursera, or YouTube for orbital mechanics and satellite communications.
3. **Explore**: Online tools like Heavens-Above to track live satellite positions.

## Zero to Hero Lab Projects
- **Satellite Ground Station**: Build a DIY ground station to track and receive data from satellites.
- **Orbital Decay Model**: Create a simulation to model the effects of atmospheric drag on low Earth orbit satellites.
- **Remote Sensing Project**: Use open-source satellite data to analyze environmental changes, like deforestation.

## Continuous Learning Strategy
1. **Next Steps**: Dive into satellite data analysis or explore mission design using software like STK.
2. **Related Topics**: Study orbital mechanics, space law, and radio frequency (RF) communication.
3. **Further Reading**: Review NASA and ESA resources for insights into real-world satellite missions.

## References
- "Satellite Communications" by Dennis Roddy.
- NASA’s Mission and Spacecraft Library.
- European Space Agency (ESA) website on satellites and missions.
