void isr (void) {
	asm {
		DI ; disable interrupts
		PUSH AF ; save registers
		PUSH BC
		PUSH DE
		PUSH HL
		PUSH IX
		PUSH IY
	}
	/* normal C code here */
	asm {
		POP IY
		POP IX
		POP HL
		POP DE
		POP BC
		POP AF
		EI ; enable interrupts
		RETI ; return from interrupt
	}
}

//some compilers interrupt indication
//interrupt void isr(void) {
/* normal C code here */
//};

// GNU GCC Compiler interrupt handler using the syntax
// void isr(void) __attribute__ ((interrupt ("IRQ"));
// void isr(void){
/* normal C code here */
//}