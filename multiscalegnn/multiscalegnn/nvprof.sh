#!/bin/bash

NVPROF="nvprof"
NVPROF_FLAGS=" -s --profile-child-processes  --profile-from-start off "
#NVPROF_FLAGS=" --profile-child-processes  --profile-from-start off --print-api-trace "

PHASE=${PHASE:-"forward"}
EPOCHS="2" 
PROFILE_EPOCHS="1"
PROFILE_BATCHES="100-110"

PROF_ROOT="./profile"
LOG_ROOT="./log"

DATE=`date '+%Y_%m_%d__%H_%M_%S'`
LOG_DIR="$LOG_ROOT/$DATE"
NVPROF_LOG="$PROF_ROOT/$DATE/"
PROF_DIR="$PROF_ROOT/$DATE/"

#METRICS="issued_ipc,issue_slot_utilization,achieved_occupancy,l2_utilization,stall_constant_memory_dependency,stall_exec_dependency,stall_inst_fetch,stall_memory_dependency,stall_memory_throttle,stall_not_selected,stall_other,stall_pipe_busy,stall_sync"
#METRICS="issued_ipc,issue_slot_utilization,achieved_occupancy,eligible_warps_per_cycle,gld_efficiency,gst_efficiency,shared_efficiency,warp_execution_efficiency,gld_throughput,gst_throughput,dram_utilization,cf_fu_utilization,double_precision_fu_utilization,half_precision_fu_utilization,issue_slot_utilization,l2_utilization,ldst_fu_utilization,shared_utilization,single_precision_fu_utilization,special_fu_utilization,sysmem_read_utilization,sysmem_utilization,sysmem_write_utilization,tex_fu_utilization,tex_utilization,stall_constant_memory_dependency,stall_exec_dependency,stall_inst_fetch,stall_memory_dependency,stall_memory_throttle,stall_not_selected,stall_other,stall_pipe_busy,stall_sync"
METRICS="flop_count_dp,flop_count_sp,flop_count_hp,inst_fp_64,inst_fp_32,inst_fp_16,inst_integer,gld_requested_throughput,gst_requested_throughput,shared_load_throughput,shared_store_throughput,gld_efficiency,gld_requested_throughput,gld_throughput,gst_efficiency,gst_requested_throughput,gst_throughput,shared_efficiency,global_hit_rate,global_load_requests,local_hit_rate,local_load_requests,local_store_requests,l2_tex_hit_rate,l2_tex_read_transactions,l2_tex_write_transactions,shared_load_transactions,shared_store_transactions,tex_cache_hit_rate,tex_cache_transactions,dram_read_transactions,dram_write_transactions"

NVPROF_METRICS_FLAGS="  --log-file $PROF_DIR/metrics-nvprof.%p.log --export-profile $PROF_DIR/train-metrics.%p.nvprof  --metrics $METRICS "
NVPROF_TIMELINE_FLAGS=" --log-file $PROF_DIR/timeline-nvprof.%p.log  --export-profile $PROF_DIR/train-timeline.%p.nvprof"

COMMAND="python snap.py "
MAIN_FLAGS=" \
    --data_dir ../data/ \
    --graph dblp \
    --maxepoch $EPOCHS \
    --profile-epoch-list $PROFILE_EPOCHS \
    --profile-batch-list $PROFILE_BATCHES \
  "

MAIN_METRICS_FLAGS=""
MAIN_TIMELINE_FLAGS=""


# Check args
if [ $# -eq 0 ]; then
  MODE='both'
else 
  MODE="$1"
fi
echo $MODE
if [ $MODE != 'both' ] && [ $MODE != 'timeline' ] && [ $MODE != 'metrics' ] ; then
  echo "Bad argument"
  echo "USAGE: $0 [both | timeline | metrics ]"
  exit 1
fi

# Add flags for profiling phase
if [ $PHASE == 'forward' ] ; then
  MAIN_FLAGS=" $MAIN_FLAGS --prof-forward "
fi
if [ $PHASE == 'backprop' ] ; then
  MAIN_FLAGS=" $MAIN_FLAGS --prof-backprop "
fi

# Create the profiling dir if it doesn't exist
if [ ! -d $PROF_DIR ] ; then
  echo "Did not find profile directory '$PROF_DIR'. Creating."
  mkdir -p $PROF_DIR
fi

# Create the logging dir if not exist
if [ ! -d $LOG_DIR ] ; then
  echo "Did not find log directory '$LOG_DIR'. Creating."
  mkdir -p $LOG_DIR
fi
if [ ! -d $LOG_DIR/timeline ] ; then
  echo "Did not find log directory '$LOG_DIR/timeline'. Creating."
  mkdir -p $LOG_DIR/timeline
fi
if [ ! -d $LOG_DIR/metrics ] ; then
  echo "Did not find log directory '$LOG_DIR/metrics'. Creating."
  mkdir -p $LOG_DIR/metrics
fi


if [ $MODE == 'both' ] || [ $MODE == 'timeline' ] ; then
  echo Generating timeline
  $NVPROF $NVPROF_FLAGS  $NVPROF_TIMELINE_FLAGS $COMMAND  $MAIN_FLAGS $MAIN_TIMELINE_FLAGS
fi

if [ $MODE == 'both' ] || [ $MODE == 'metrics' ] ; then
  echo Gathering Metrics
  $NVPROF $NVPROF_FLAGS  $NVPROF_METRICS_FLAGS  $COMMAND  $MAIN_FLAGS $MAIN_METRICS_FLAGS
fi

# link latest dirs
(
  rm -f latest
  cd $PROF_ROOT
  ln -f -s $DATE -T latest
)

