
int AcmeO2Adapter_gimmeO2Conc(AcmeO2Adapter* const me) {
	return me->itsAcmeO2SensorProxy->getO2Conc();
}

int AcmeO2Adapter_gimmeO2Flow(AcmeO2Adapter* const me) {
	return (me->itsAcmeO2SensorProxy->getO2Flow()*60)/100;
}