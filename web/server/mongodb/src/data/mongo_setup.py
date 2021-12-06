import mongoengine

def global_init():
    # multiples connections/databases
    mongoengine.register_connection(alias='core', name='mongo_test')