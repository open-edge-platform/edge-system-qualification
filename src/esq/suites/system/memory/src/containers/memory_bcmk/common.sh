#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# set -x

ch=("|" "\\" "-" "/")
time="date \"+%H:%M:%S\".\$((10#\$(date \"+%N\")/1000000))"

baseFileName=$(basename "$0")

_red=$(tput setaf 1);_green=$(tput setaf 2);_yellow=$(tput setaf 3);_magenta=$(tput setaf 5);_reset=$(tput sgr0)

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


# Function to check if Docker daemon is running
function check_docker_daemon {
  detailsLogFile=$1
  if ! systemctl status docker &> /dev/null; then
    format_log "Docker daemon is not running. Please start Docker and try again." "1" "${FUNCNAME[0]}" "$LINENO" "$detailsLogFile"
    exit 1
  fi
  format_log "Docker daemon is running." "0" "${FUNCNAME[0]}" "$LINENO" "$detailsLogFile"
}

# Function to check if user has permission to run Docker commands
function check_docker_permission {
  detailsLogFile=$1
  if ! groups "$(whoami)" | grep &>/dev/null '\bdocker\b'; then
    format_log "You do not have permission to run Docker commands. Please add yourself to the 'docker' group and try again." "1"  "${FUNCNAME[0]}" "$LINENO" "$detailsLogFile"
    exit 1
  fi
  format_log "Checked you have permission to run Docker commands." "0" "${FUNCNAME[0]}" "$LINENO" "$detailsLogFile"
}

# Function to convert string like '256K' to actual byte number.
_convert_to_bytes() {
  input=$1
  suffix=${input: -1}   # Extract the last character from the input
  local result=-1

  if [ "$suffix" == "K" ] || [ "$suffix" == "k" ]; then
      value=${input%?}   # Remove the last character from the input
      result=$((value * 1024))
  elif [ "$suffix" == "M" ] || [ "$suffix" == "m" ]; then
      value=${input%?}
      result=$((value * 1024 * 1024))
  else
      value=${input}
      re='^[0-9]+$'
      if ! [[ $value =~ $re ]] ; then
          echo "Invalid input. Please provide a valid number followed by 'K' or 'M'."
      else
          result=${value}
      fi
  fi

  echo "$result"
}

# Function to check the L3 cache size, then return the STREAM_ARRAY_SIZE size which more than 4 times of cache size.
get_stream_array_size() {
  default_val=120000000
  #l3_cache_result=($(lscpu |grep 'L3 cache'))
  l3_cache_result=()
  IFS=" " read -r -a l3_cache_result <<< "$(getconf LEVEL3_CACHE_SIZE)"
  [ ${#l3_cache_result[@]} -eq 0 ] && return ${default_val}
  l3_cache_byte=${l3_cache_result[-1]}
  if [ "${l3_cache_byte}" -ne -1 ];then
    echo $((5 * l3_cache_byte)) # so we use 5 times of cache size
  else
    echo ${default_val}
  fi
}
