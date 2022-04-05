==================== NOTES =======================

Buffer can be implemented in HW and SW. 
-> Helps >>transmitting<< lots of data IN or OUT of an interface(eg UART)

--> Buf_add()
--> buf_remove()


=> Circular buffer (Ring buffer) is an exemple of a FIFO (First In First Out) buffer.(Without END!!!!)
	-> Create with a contigous(one closed to another) block of memory
	
-> Adds Data to one end(Head)
-> Removes data items from the other end (Tail)

***HEAD : number of items ADDED!!!
***TAIL : number of items REMOVED!!!





Lien : https://en.wikipedia.org/wiki/Circular_buffer

https://www.coursera.org/lecture/embedded-software-hardware/7-circular-buffer-7DqTE

http://www.martinbroadhurst.com/cirque-in-c.html

/!\ https://blog.stratifylabs.co/device/2013-10-02-A-FIFO-Buffer-Implementation/


