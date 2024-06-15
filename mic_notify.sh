#!/bin/bash

# vim /etc/udev/rules.d/99-usb-mic-notify.rules
# ACTION=="add", SUBSYSTEM=="sound", KERNEL=="card*", RUN+="/opt/dorylus/mic_notify.sh $env{ACTION} $env{SUBSYSTEM} $env{DEVPATH} $env{DEVNAME} $env{ID_MODEL} $env{ID_VENDOR}"
# ACTION=="remove", SUBSYSTEM=="sound", KERNEL=="card*", RUN+="/opt/dorylus/mic_notify.sh $env{ACTION} $env{SUBSYSTEM} $env{DEVPATH} $env{DEVNAME} $env{ID_MODEL} $env{ID_VENDOR}"

action=$1
sub_system=$2
dev_path=$3
dev_name=$4
id_model=$5
id_vendor=$6

cur_user=`/usr/bin/whoami`

lock_file="/var/lock/.mic-notify-devices.lock"

(
    flock -w 10 200||exit 1
    case "$action" in
    add )
        echo "$cur_user: $action, $sub_system, $dev_path, $dev_name, $id_model, $id_vendor" >> /opt/dorylus/logs/udev.log
        /usr/bin/aplay /opt/dorylus/cache/sounds/plug_sound.wav
        sleep 3

        # restart pulseaudio
        /usr/bin/killall pulseaudio
        /usr/bin/pulseaudio --daemonize=true --system --disallow-exit --disallow-module-loading
        ;;
    remove )
        echo "$cur_user: $action, $sub_system, $dev_path, $dev_name, $id_model, $id_vendor" >> /opt/dorylus/logs/udev.log
        /usr/bin/aplay /opt/dorylus/cache/sounds/unplug_sound.wav
        sleep 3

        # restart pulseaudio
        /usr/bin/killall pulseaudio
        /usr/bin/pulseaudio --daemonize=true --system --disallow-exit --disallow-module-loading
        ;;
    esac
) 200>"$lock_file"
