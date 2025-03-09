#include <time.h>
#include <stdbool.h>

typedef struct {
    time_t timeout;
    time_t last_feed;
    bool enabled;
} Watchdog;

void wd_init(Watchdog* wd, unsigned seconds) {
    wd->timeout = seconds;
    wd->enabled = false;
}

void wd_feed(Watchdog* wd) {
    wd->last_feed = time(NULL);
}

void wd_enable(Watchdog* wd) {
    wd->enabled = true;
    wd_feed(wd);
}

bool wd_check(Watchdog* wd) {
    return wd->enabled && (time(NULL) - wd->last_feed > wd->timeout);
}


int main(void)
{
    Watchdog wd;
    wd_init(&wd, 5);
    wd_enable(&wd);
    
    while(true) {
        if(wd_check(&wd)) {
            printf("Watchdog timeout\n");
            break;
        }
        wd_feed(&wd);
    }
    
    return 0;
}