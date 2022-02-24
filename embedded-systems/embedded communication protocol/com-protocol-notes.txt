# embedded communication protocols notes

## OSI Model Layer 
+------------+--------------+------------------------+--------------------------+ 
						LAYER						 | Protocol Data Unit (PDU) |
+------------+--------------+---+--------------------+--------------------------+
| 			 | 				| 7	| Application layer  | 							|
+ 			 |				+---+--------------------+							+
| 			 | 				| 6	| Presentation layer | 		     DATA			|
+ High Layer | Application  +---+--------------------+							+
| 			 |	Layer	    | 5	|	Session layer	 |							|
+			 +--------------+---+--------------------+---------------------------
| 			 | 				| 4 |	Transport layer  | 		Segment, datagram   |
+------------|				+---+--------------------+---------------------------
|			 | 				| 3 |	Network layer	 |	    Packet			    |
+		 	 |	Data flow	+---+--------------------+---------------------------
| HW Layer	 |	Layer		| 2	| Data link layer    | 		Frame			    |
+ 			 |				+---+--------------------+---------------------------
| 			 | 				| 1	|	Physical layer   |	 	Bit, Symbol		    |
+------------+--------------+------------------------+--------------------------+
                                                       
## List of communication protocols

- SPI/QSPI
- I2C
- UART/USART
- BLUETHOOT
- USB
- ETHERNET
- CAN/CANFD
- FLEXRAY

STANDARDs
- J1939

# References : 

communication protocols : https://en.wikipedia.org/wiki/Communication_protocol#Layering
OSI Model : https://en.wikipedia.org/wiki/OSI_model
