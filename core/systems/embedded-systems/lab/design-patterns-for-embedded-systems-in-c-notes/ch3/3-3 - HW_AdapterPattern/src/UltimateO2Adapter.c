

int UltimateO2Adapter_gimmeO2Conc(UltimateO2Adapter* const me) {
	return int(me->getItsUltimateO2SensorProxy->accessO2Conc()*100);
}

int UltimateO2Adapter_gimmeO2Flow(UltimateO2Adapter* const me) {
	double totalFlow;
	// convert from liters/hr to cc/min
	totalFlow = me->itsUltimateO2SensorProxy->accessGasFlow() * 1000.0/60.0;
	// now return the portion of the flow due to oxygen
	// and return it as an integer
	return (int)(totalFlow * me->itsUltimateO2SensorProxy->accessO2Conc());
}