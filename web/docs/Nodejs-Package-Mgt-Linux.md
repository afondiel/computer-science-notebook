# Nodejs and Module Management on Linux


## Overview 

This is a setup for a tiny HMI project built in js running on Raspberry Pi.

- HMI goals is display a data stream from a sensor.
- The Nodejs engine and PM2 help to speed the data transmission and display on the UI. 


## Node Installation & Dependencies

Check if the version is the last LTS one

```sh
curl -sL https://deb.nodesource.com/setup_10.x | sudo -E bash -
sudo apt install -y nodejs
```

Check the ```node.js``` & ```npm``` version

```
node -v

&&

npm -v
```

## Install build-essential package

```sh
sudo apt install build-essential
```

## Install Process Manager (PM2)

```sh
sudo npm install pm2@latest -g
```

**Warning:** 
- Each time you reboot your machine you have to start the server!**
- Make the server load when the Pi reboots

```sh
pm2 startup
```

## HMI 

The HMI has two modules

- Server Module `server.js`: handle the PM2 engine
- Client Mudule `clien.js`: handle Hmi frontend configuration

To start the server

```sh
pm2 start server.js
```

Save the state of the server to be launched after the next reboot of the system : 

```sh
pm2 save
```

Reboot the system to take the effect

```sh
reboot
```

## PM2 Usages

- Start the server (with watch)

```sh 
pm2 start server.js --watch 
```

- Restart the server (without watch)

```sh 
pm2 stop server && pm2 start server.js 
```

- Monitor the server

```sh 
pm2 monit server 
```

- List of server launch with PM2

```sh 
pm2 list
``` 
or 

```sh 
pm2 ls 
```

## Install Nodejs packages modules both for the client and the server

Go into the client folder and run 

```sh 
npm install
```

Run the npm package for the `client`

```sh
npm start
``` 

Go into the `server` folder and run 

```sh 
npm install
```

## Local Dev 

```sh
npm run dev
```

```
https://localhost:3000
``` 


## Deployement => Production

- How to build the client files after a modification or an update
- build the new modification and this will generate new setup files for the HMI

```sh
npm run build
```

## References

- [PM2 (software)](https://en.wikipedia.org/wiki/PM2_(software))
- [PM2 - Project](https://pm2.keymetrics.io/docs/usage/quick-start/)
- [NVM - version manager for node.js](https://github.com/nvm-sh/nvm)

