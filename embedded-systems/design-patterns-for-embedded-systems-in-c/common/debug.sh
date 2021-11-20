#!/bin/sh

SCRIPTS=/c/Ac6/SystemWorkbench/plugins/fr.ac6.mcu.debug_1.11.1.201611241417/resources/openocd/scripts
TARGET=/c/Ac6/SystemWorkbench/plugins/fr.ac6.mcu.debug_1.11.1.201611241417/resources/openocd/scripts/target/stm32l1.cfg
BIN=/c/Ac6/SystemWorkbench/plugins/fr.ac6.mcu.externaltools.openocd.win32_1.12.0.201611241417/tools/openocd/bin
CFG=/c/Users/mors/Documents/git/riot/boards/unwd-range-l1-r2/dist/umdk-jtag.cf

cd $SCRIPTS
$BIN/openocd.exe -f $CFG -f $TARGET