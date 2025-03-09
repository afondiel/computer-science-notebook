# RTOS in Industry

## Automotive
RTOSes commonly used or certified for automotive systems, often compliant with **ISO 26262** (functional safety standard for automotive).

| RTOS Name       | Developer                   | License         | Certification Standards | Notes                                      |
|-----------------|-----------------------------|-----------------|-------------------------|--------------------------------------------|
| **QNX**         | BlackBerry                  | Proprietary     | ISO 26262, IEC 61508    | Widely used in automotive infotainment and ADAS. |
| **VxWorks**     | Wind River                  | Proprietary     | ISO 26262, DO-178C      | Deployed in automotive and safety-critical systems. |
| **INTEGRITY**   | Green Hills Software        | Proprietary     | ISO 26262, DO-178B      | High-security RTOS for automotive ECUs.    |
| **ThreadX**     | Microsoft (via Express Logic) | Proprietary   | ISO 26262, IEC 61508    | Small footprint, used in automotive controllers. |
| **PikeOS**      | SYSGO                       | Proprietary     | ISO 26262, EN 50128     | Hypervisor/RTOS for automotive and rail.  |

---

## Avionics
RTOSes certified or used in avionics, often compliant with **DO-178B/C** (software safety standard for airborne systems).

| RTOS Name       | Developer                   | License         | Certification Standards | Notes                                      |
|-----------------|-----------------------------|-----------------|-------------------------|--------------------------------------------|
| **VxWorks**     | Wind River                  | Proprietary     | DO-178C, ISO 26262      | Industry standard for avionics systems.    |
| **INTEGRITY**   | Green Hills Software        | Proprietary     | DO-178B, ISO 26262      | Used in safety-critical avionics.          |
| **LynxOS**      | Lynx Software Technologies  | Proprietary     | DO-178B                 | POSIX-compliant, common in avionics.       |
| **PikeOS**      | SYSGO                       | Proprietary     | DO-178C, EN 50128       | Certified for avionics and rail systems.   |
| **uC/OS-II**    | Micrium (Silicon Labs)      | Proprietary     | DO-178B (via kit)       | Simple RTOS with certification support.    |
| **uC/OS-III**   | Micrium (Silicon Labs)      | Proprietary     | DO-178B, IEC 61508      | Enhanced version for avionics use.         |

---

## Industrial
RTOSes used in industrial automation, often compliant with **IEC 61508** (functional safety for industrial systems).

| RTOS Name       | Developer                   | License         | Certification Standards | Notes                                      |
|-----------------|-----------------------------|-----------------|-------------------------|--------------------------------------------|
| **QNX**         | BlackBerry                  | Proprietary     | IEC 61508, ISO 26262    | Used in industrial control systems.        |
| **ThreadX**     | Microsoft (via Express Logic) | Proprietary   | IEC 61508, ISO 26262    | Compact RTOS for industrial devices.       |
| **uC/OS-III**   | Micrium (Silicon Labs)      | Proprietary     | IEC 61508, DO-178B      | Suitable for industrial automation.        |
| **FreeRTOS**    | Amazon (Real Time Engineers) | MIT            | IEC 61508 (via SafeRTOS)| SafeRTOS variant for industrial safety.    |
| **INTEGRITY**   | Green Hills Software        | Proprietary     | IEC 61508, DO-178B      | High-reliability industrial applications.  |

---

## Internet of Things (IoT)
RTOSes designed for resource-constrained devices in IoT applications.

| RTOS Name       | Developer                   | License         | Certification Standards | Notes                                      |
|-----------------|-----------------------------|-----------------|-------------------------|--------------------------------------------|
| **FreeRTOS**    | Amazon (Real Time Engineers) | MIT            | IEC 61508 (via SafeRTOS)| Lightweight, dominant in IoT ecosystems.   |
| **Zephyr**      | Linux Foundation            | Apache 2.0      | -                       | IoT-focused, supports constrained devices. |
| **NuttX**       | Apache Foundation           | Apache 2.0      | -                       | POSIX-compliant, used in IoT hardware.     |
| **ThreadX**     | Microsoft (via Express Logic) | Proprietary   | IEC 61508, ISO 26262    | Small footprint, suitable for IoT sensors. |
| **ChibiOS/RT**  | Giovanni Di Sirio           | GPL3 / Commercial | -                     | Lightweight, hobbyist IoT projects.        |

---

## Aerospace
RTOSes used in aerospace applications, often overlapping with avionics but including broader space-specific uses.

| RTOS Name       | Developer                   | License         | Certification Standards | Notes                                      |
|-----------------|-----------------------------|-----------------|-------------------------|--------------------------------------------|
| **RTEMS**       | RTEMS Project               | BSD             | -                       | Open-source, used in space missions (e.g., NASA). |
| **VxWorks**     | Wind River                  | Proprietary     | DO-178C, ISO 26262      | Deployed in aerospace and satellite systems. |
| **INTEGRITY**   | Green Hills Software        | Proprietary     | DO-178B, ISO 26262      | Safety-critical aerospace applications.    |
| **PikeOS**      | SYSGO                       | Proprietary     | DO-178C, EN 50128       | Used in aerospace and defense.             |

---

## General-Purpose / Others
RTOSes without specific industry certifications but versatile across multiple domains.

| RTOS Name       | Developer                   | License         | Certification Standards | Notes                                      |
|-----------------|-----------------------------|-----------------|-------------------------|--------------------------------------------|
| **eCos**        | eCosCentric / Community     | eCos License    | -                       | Configurable, general embedded systems.    |
| **Fuchsia (Zircon)** | Google                  | BSD/MIT/Apache 2.0 | -                    | Microkernel, experimental RTOS capabilities. |
| **uC/OS-II**    | Micrium (Silicon Labs)      | Proprietary     | DO-178B (via kit)       | Educational and simple embedded systems.   |

---

## Notes
- **Certification Standards**: 
  - **ISO 26262**: Automotive functional safety.
  - **DO-178B/C**: Avionics software safety.
  - **IEC 61508**: General industrial functional safety.
  - **EN 50128**: Railway systems (noted for PikeOS).
  - "-" indicates no widely recognized industry-specific certification.
- **Overlap**: Some RTOSes (e.g., VxWorks, QNX) are used across multiple industries due to their versatility and certifications.
- **IoT**: While not always certified, these RTOSes are optimized for low-power, connected devices.
