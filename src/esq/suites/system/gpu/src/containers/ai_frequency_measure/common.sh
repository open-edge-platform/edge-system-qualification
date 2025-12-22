#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ch=("|" "\\" "-" "/")
time="date \"+%H:%M:%S\".\$((10#\$(date \"+%N\")/1000000))"

baseFileName=$(basename "$0")
# Check if terminal is available for color output
if [ -t 1 ] && command -v tput >/dev/null 2>&1 && tput colors >/dev/null 2>&1; then
    _red=$(tput setaf 1);_green=$(tput setaf 2);_yellow=$(tput setaf 3);_magenta=$(tput setaf 5);_reset=$(tput sgr0)
else
    _red="";_green="";_yellow="";_magenta="";_reset=""
fi

iGPU_Dev_IDs=(A7A9 A7A8 A7A1 A7A0 A721 A720 A78B A78A A789 A788 A783 A782 A781 A780 4907 4905 4680 4682 4688 468A 468B 4690 4692 4693 46D0 46D1 46D2 4626 4628 462A 46A0 46A1 46A2 46A3 46A6 46A8 46AA 46B0 46B1 46B2 46B3 46C0 46C1 46C2 46C3 4C8A 4C8B 4C90 4C9A 4C8C 4C80 4E71 4E61 4E57 4E55 4E51 4571 4557 4555 4551 4541 9A59 9A60 9A68 9A70 9A40 9A49 9A78 9AC0 9AC9 9AD9 9AF8 6420 64B0 7D51 7D67 7D41 7DD1 7DD5 7D45 7D40 7D55 0BD5 0BDA 56C0 56C1 A7AA A7AB A7AC A7AD 4908 4909 46D3 46D4)

dGPU_Dev_IDs=(56B3 56B2 56A4 56A3 56BA 5697 5696 5695 56B1 56B0 56A6 56A5 56A1 56A0 5694 5693 5692 5691 5690 E20B E20C 64A0 7D55 56A2 56BC 56BD 56BB E211 E212)

find_available_gpus()
{
  devices=$(lspci -nn | grep -Ei "DISPLAY|VGA")

  # Extract vendor and device IDs from the output
  vendor_ids=()
  device_ids=()
  available_devices=()

  while IFS=" " read -r id_info; do
    vendor_id=$(echo "$id_info" | sed -n 's/.*\[\([0-9]*\):.*/\1/p')
    device_id=$(echo "$id_info" | sed -n 's/.*:\([[:alnum:]]*\)].*/\1/p')
    if [[ "${iGPU_Dev_IDs[*]}" =~ ${device_id^^} ]]; then
	available_devices+=("iGPU")
     elif [[ "${dGPU_Dev_IDs[*]}" =~ ${device_id^^} ]]; then
	available_devices+=("dGPU")
     fi 
    vendor_ids+=("$vendor_id")
    device_ids+=("$device_id")
  done <<< "$devices"
}

format_log()
{
    body=$1
    returnCode=$2
    funcName=$3
    lineNo=$4
    detailsLogFile=$5
    if [ -z "${returnCode}" ];then
        echo "[$(eval "$time")][$baseFileName][$funcName][line:$lineNo]${body}" 2>&1 | tee -a "$detailsLogFile"
    elif [ "${returnCode}" -eq 0 ]; then
        echo "[$(eval "$time")][$baseFileName][$funcName][line:$lineNo]${_green}${body}${_reset}" 2>&1 | tee -a "$detailsLogFile"
    else
        echo "[$(eval "$time")][$baseFileName][$funcName][line:$lineNo]${_red}${body}${_reset}" 2>&1 | tee -a "$detailsLogFile"
    fi
}

format_log_np()
{
    body=$1
    returnCode=$2
    funcName=$3
    lineNo=$4
    detailsLogFile=$5
    echo "[$(eval "$time")][$baseFileName][$funcName][line:$lineNo]${body}[return code: $returnCode]" >> "$detailsLogFile" 2>&1
}

format_log_section()
{
    body=$1
    returnCode=$2
    funcName=$3
    lineNo=$4
    detailsLogFile=$5
    len=${len:-0}
    splitStr=$(printf '%0.s-' $(seq 1 $((120 - len > 0 ? 120 - len : 0))))

    echo -e "\n"
    echo "[$(eval "$time")][$baseFileName][$funcName][line:$lineNo]${splitStr}" | tee -a "$detailsLogFile"
    echo "[$(eval "$time")][$baseFileName][$funcName][line:$lineNo]${_yellow}${body}${_reset}" 2>&1 | tee -a "$detailsLogFile"
    echo "[$(eval "$time")][$baseFileName][$funcName][line:$lineNo]${splitStr}" | tee -a "$detailsLogFile"
}

checkProcessStatus()
{
    monitoring=$(pgrep -c "$2")
    return "$monitoring"
}

barFormat()
{
    index=0
    APPPID=$1
    testApplicationName=$2
    testModuleName=$3
    testCaseName=$4
    funcName=$5
    lineNo=$6
    detailsLogFile=$7
    string=${baseFileName}${funcName}${lineNo}${testModuleName}${testCaseName}
    len=${#string}
    splitStr=$(printf '%0.s-' $(seq 1 $((120 - len))))

    checkProcessStatus "$testApplicationName" "$APPPID"
    retCode=$?
    while [ $retCode -ge 1 ]
    do
        retCode=0

	printf "[%-12s][%s][%s][%s][%s][%s]%s[%c]\r" "$(eval "$time")" "${baseFileName}" "${funcName}" "${lineNo}" "${testModuleName}" "${testCaseName}" "${splitStr}" "${ch[$index]}"
	((index++))
	(( index = index % 4 ))
        for ((i=0;i<5;i++))
        do
            checkProcessStatus "$testApplicationName" "$APPPID"
            retCode=$?
            # retCode=`expr $retCode + $retTemp`
            sleep 0.1
        done
    done
    wait "$APPPID"
    returnCode=$?
    if [ $returnCode -eq 0 ]
    then
	printf "[%-12s][%s][%s][%s][%s][%s]%s[%s]\n" "$(eval "$time")" "$baseFileName" "$funcName" "$lineNo" "$testModuleName" "$testCaseName" "$splitStr" "${_green}Pass${_reset}" 2>&1 | tee -a "$detailsLogFile"

    else
	printf "[%-12s][%s][%s][%s][%s][%s]%s[%s]\n" "$(eval "$time")" "$baseFileName" "$funcName" "$lineNo" "$testModuleName" "$testCaseName" "$splitStr" "${_red}Fail${_reset}" 2>&1 | tee -a "$detailsLogFile"
    fi

    return $returnCode
}


