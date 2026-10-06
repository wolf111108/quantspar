"""
Statistics Manager for Quantization Calibration.

This module provides classes for collecting and managing quantization statistics
during the calibration phase using PyTorch hooks.
"""
import math
import os
from os import path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from collections import defaultdict
from .quant_spec import QuantSpec, parse_quant_spec
from .cim_stats import sparse_counts, measure_mapping, unit_sparse_counts
from typing import Optional, Tuple, Dict
from .mapping import CIM_sys, Mapping_stat, sy_latency, Mapping_stat_dynamic

class QuantStatistics:
    """
    Statistics collector for a single layer.
    
    Collects scale information for activations, weights, and outputs
    during calibration.
    """
    
    def __init__(self, layer_name: str, layer_idx: int):
        """
        Initialize statistics collector.
        
        Args:
            layer_name: Name of the layer (e.g., 'q_proj', 'k_proj')
            layer_idx: Index of the layer in the model
        """
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        
        # Scale statistics
        self.w_scales: List[float] = []
        self.a_scales: List[float] = []
        self.o_scales: List[float] = []
        
        # For matmul layers
        self.A_scales: List[float] = []
        self.B_scales: List[float] = []
        
        # Sample count
        self.sample_count = 0
    
    def collect_linear_stats(
        self, 
        w_scale: float, 
        a_scale: float, 
        o_scale: float
    ):
        """
        Collect statistics for linear layer.
        
        Args:
            w_scale: Weight scale
            a_scale: Activation scale
            o_scale: Output scale
        """
        self.w_scales.append(w_scale)
        self.a_scales.append(a_scale)
        self.o_scales.append(o_scale)
        self.sample_count += 1
    
    def collect_matmul_stats(
        self,
        A_scale: float,
        B_scale: float,
        O_scale: float
    ):
        """
        Collect statistics for matrix multiplication.
        
        Args:
            A_scale: First input scale
            B_scale: Second input scale
            O_scale: Output scale
        """
        if A_scale is not None:
            self.A_scales.append(A_scale)
        if B_scale is not None:
            self.B_scales.append(B_scale)
        self.o_scales.append(O_scale)
        self.sample_count += 1
        if self.layer_idx == 0:
            if self.layer_name =="qk_matmul":
                pass


    def get_final_scales(self) -> Dict[str, float]:
        """
        Compute final scales from collected statistics.
        
        Uses maximum value across all samples for robustness.
        
        Returns:
            Dictionary of final scales
        """
        scales = {}
        
        if self.w_scales:
            if isinstance(self.w_scales[0], torch.Tensor):
                scales['w_scale'] = torch.stack(self.w_scales).amax(dim=0)
            else:
                scales['w_scale'] = max(self.w_scales)
            scales['a_scale'] = max(self.a_scales)
            scales['o_scale'] = max(self.o_scales)
        
        if self.A_scales:
            scales['A_scale'] = max(self.A_scales)
            scales['B_scale'] = max(self.B_scales)
            # For matmul, O_scale is stored in o_scales list
            if self.o_scales:
                scales['O_scale'] = max(self.o_scales)
        
        return scales
    
    def reset(self):
        """Reset all collected statistics."""
        self.w_scales.clear()
        self.a_scales.clear()
        self.o_scales.clear()
        self.A_scales.clear()
        self.B_scales.clear()
        self.sample_count = 0


class QuantStatManager:
    """
    Global statistics manager for quantization calibration.

    Manages statistics collection for all layers in the model using
    PyTorch hooks.
    """

    def __init__(self, scale_dir: str, nmacro: int = 32, as_l: int = 1, *,
                 h: int = 64, w: int = 48, banks: int = 16,
                 bit_scope: str = "mantissa", cycles_per_effective_bit: float = 1,
                 ebb_config=None):
        """
        Initialize statistics manager.

        Args:
            scale_dir: Directory to save/load scales
            nmacro: Number of macro units
            as_l: Mapping_stat_dynamic sync model (1=async, 0=sync)
        """
        self.scale_dir = scale_dir
        self.nmacro = int(nmacro)
        self.as_l = int(as_l)
        if self.as_l not in (0,1): raise ValueError("as_l must be 0 or 1")
        if bit_scope not in ("mantissa","sign_mantissa","storage"): raise ValueError("invalid bit_scope")
        if min(h,w,banks,self.nmacro)<=0: raise ValueError("CIM geometry must be positive")
        if not math.isfinite(cycles_per_effective_bit) or cycles_per_effective_bit<=0:
            raise ValueError("cycles_per_effective_bit must be finite and positive")
        self.cim_geometry = dict(height=int(h),width=int(w),banks=int(banks),macros=self.nmacro)
        self.bit_scope = bit_scope
        self.cim_records = []
        self.ebb_stats = None
        self.execution_context = {}
        self.attention_context = {}
        if ebb_config is not None and ebb_config.get("enabled", True):
            from .ebb import EBBConfig, EBBStats
            self.ebb_stats = EBBStats(EBBConfig.from_dict(ebb_config), ebb_config.get("trace_path"))
        self._static_weights_seen = set()
        self.stats: Dict[str, QuantStatistics] = {}
        self.hooks: List[Any] = []

        # Create scale directory if it doesn't exist
        os.makedirs(scale_dir, exist_ok=True)

        self.total_zero_count = 0
        self.total_element_count = 0
        self.total_bit_count = 0
        self.total_0bit_count = 0
        self.total_sparsebit_count = 0
        self.total_amplitude_zero_bits_total = 0
        self.total_amplitude_bit_count = 0  #add

        self.activation_zero_count = 0
        self.activation_element_count = 0
        self.activation_bit_count = 0
        self.activation_0bit_count = 0
        self.activation_sparsebit_count = 0
        self.activation_amplitude_zero_bits_total = 0
        self.activation_amplitude_bit_count = 0  #add


        self.shift_bit_total_count = 0
        self.shift_bit_0_count = 0


        self.weight_zero_count = 0
        self.weight_element_count = 0
        self.weight_bit_count = 0
        self.weight_0bit_count = 0
        self.weight_sparsebit_count = 0
        self.weight_amplitude_zero_bits_total = 0
        self.weight_amplitude_bit_count = 0  #add

        self.dynamic_weight_zero_count = 0
        self.dynamic_weight_element_count = 0
        self.dynamic_weight_bit_count = 0
        self.dynamic_weight_0bit_count = 0
        self.dynamic_weight_sparsebit_count = 0
        self.dynamic_weight_amplitude_zero_bits_total = 0
        self.dynamic_weight_amplitude_bit_count = 0  #add

        self.q_proj_mapping_stat = []
        self.qk_matmul_A_mapping_stat = []
        self.qk_matmul_B_mapping_stat = []
        self.pv_matmul_A_mapping_stat = []
        self.pv_matmul_B_mapping_stat = []
        self.o_proj_mapping_stat = []
        self.up_proj_mapping_stat = []
        self.down_proj_mapping_stat = []

        self.mapping_stat_bf16_stat_b1 = []
        self.mapping_stat_fp8_stat_b1 = []
        self.mapping_stat_int8_stat_b1 = []
        self.mapping_stat_bf16_stat_b2 = []
        self.mapping_stat_fp8_stat_b2 = []
        self.mapping_stat_int8_stat_b2 = []

        # Per-layer latency records: key = f"{layer_name}_{layer_idx}"
        self.per_layer_latency = {}

        self.SACIM_latency_stat = 0
        self.ideal_sparsity_latency_stat = 0
        self.baseline_latency_stat = 0
        self.latency = 0


        self.pv_amplitude_zero_bits_all = 0
        self.pv_amplitude_zero_bits_no_causal = 0
        self.pv_per_token_effective_one_bits_all = None  # tensor, 第一次累加时赋值
        self.pv_total_bits = 0
        self.pv_count = 0

        # Hardware timing model:
        # each effective activation bit uses the configured clock-cycle factor.
        # Mapping_stat / Mapping_stat_dynamic return effective-bit steps;
        # conversion to physical time applies this factor exactly once.
        self.clock_freq_hz = 1.0e9
        self.cycles_per_effective_bit = cycles_per_effective_bit

        self.q_proj_SACIM_latency_stat = 0
        self.q_proj_baseline_latency_stat = 0
        self.q_proj_sparsity_speedup = 0

        self.qk_matmul_SACIM_latency_stat = 0
        self.qk_matmul_baseline_latency_stat = 0
        self.qk_matmul_sparsity_speedup = 0

        self.pv_matmul_SACIM_latency_stat = 0
        self.pv_matmul_baseline_latency_stat = 0
        self.pv_matmul_sparsity_speedup = 0

        self.o_proj_SACIM_latency_stat = 0
        self.o_proj_baseline_latency_stat = 0
        self.o_proj_sparsity_speedup = 0

        self.up_proj_SACIM_latency_stat = 0
        self.up_proj_baseline_latency_stat = 0
        self.up_proj_sparsity_speedup = 0

        self.down_proj_SACIM_latency_stat = 0
        self.down_proj_baseline_latency_stat = 0
        self.down_proj_sparsity_speedup = 0

        self.gate_proj_SACIM_latency_stat = 0
        self.gate_proj_baseline_latency_stat = 0
        self.gate_proj_sparsity_speedup = 0

        self.current_phase = "full_forward"  #add
        self.phase_sparsity = {  #add
            "full_forward": self._new_sparsity_counter(),  #add
            "prefill": self._new_sparsity_counter(),  #add
            "decode": self._new_sparsity_counter(),  #add
        }  #add

        # ------------------------------------------------------------
        # Collected layer names for debug/checking  #add
        # phase -> set(layer_key)  #add
        # phase -> layer_key -> call_count  #add
        # ------------------------------------------------------------
        self.collected_layer_names_by_phase = {  #add
            "full_forward": set(),  #add
            "prefill": set(),  #add
            "decode": set(),  #add
        }  #add

        self.collected_layer_call_count_by_phase = {  #add
            "full_forward": {},  #add
            "prefill": {},  #add
            "decode": {},  #add
        }  #add

        # chunked sparse/bit statistics config  #add
        # 如果 1_048_576 仍然 OOM，可以改小到 262_144 或 131_072。 #add
        self.sparse_stat_chunk_size = 1_048_576  #add

                # ------------------------------------------------------------
        # Unit/block sparsity config  #add
        #
        # unit_bit_group_size:
        #   a，表示每个 unit 统计多少个 bit
        #
        # unit_dim_group_size:
        #   b，表示每个 unit 统计多少个维度
        #
        # enable_unit_sparsity:
        #   是否开启新的 unit 稀疏度统计
        # ------------------------------------------------------------
        self.enable_unit_sparsity = True  #add
        self.unit_bit_group_size = 2  #add
        self.unit_dim_group_size = 2  #add


        # phase -> layer_key -> counter  #add
        self.unit_sparsity = {  #add
            "full_forward": {},  #add
            "prefill": {},  #add
            "decode": {},  #add
        }  #add

    def _new_sparsity_counter(self):  #add
        return {  #add
            "total_zero_count": 0,  #add
            "total_element_count": 0,  #add
            "total_bit_count": 0,  #add
            "total_0bit_count": 0,  #add
            "total_sparsebit_count": 0,  #add
            "total_amplitude_zero_bits_total": 0,  #add
            "total_amplitude_bit_count": 0,  #add
        }  #add

    def set_phase(self, phase: str):  #add
        if phase not in self.phase_sparsity:  #add
            raise ValueError(f"Unknown sparsity phase: {phase}")  #add
        self.current_phase = phase  #add

    def set_step(self, step, cache_length_before=0, **metadata):
        """Explicit inference-call provenance; decode cache grows every step."""
        if step < 0 or cache_length_before < 0:
            raise ValueError("step and cache length must be nonnegative")
        self.execution_context = dict(step=int(step), cache_length_before=int(cache_length_before),
                                      **metadata)
        self.attention_context.clear()

    def set_attention_context(self, layer_idx, **metadata):
        for name in ("qk_matmul", "pv_matmul"):
            self.attention_context[(name, layer_idx)] = dict(metadata)

    def export_ebb_stats(self, path, workload=None):
        if self.ebb_stats is None:
            raise ValueError("EBB backend is not enabled")
        return self.ebb_stats.export(path, workload)

    def close(self):
        if self.ebb_stats is not None:
            self.ebb_stats.close()

    def reset_sparsity(self):  #add
        if self.ebb_stats is not None:
            raise ValueError("Use a new EBB manager per run; reset would invalidate streamed trace")
        self.cim_records.clear();self._static_weights_seen.clear();self.per_layer_latency.clear()
        for prefix in ('activation','weight','dynamic_weight'):
            for suffix in ('zero_count','element_count','bit_count','0bit_count','sparsebit_count',
                           'amplitude_zero_bits_total','amplitude_bit_count'):
                setattr(self,prefix+'_'+suffix,0)
        for name in list(vars(self)):
            if name.endswith(('_latency_stat','_sparsity_speedup')):
                setattr(self,name,0)
        self.latency=0
        self.total_zero_count = 0  #add
        self.total_element_count = 0  #add
        self.total_bit_count = 0  #add
        self.total_0bit_count = 0  #add
        self.total_sparsebit_count = 0  #add
        self.total_amplitude_zero_bits_total = 0  #add
        self.total_amplitude_bit_count = 0  #add

        self.phase_sparsity = {  #add
            "full_forward": self._new_sparsity_counter(),  #add
            "prefill": self._new_sparsity_counter(),  #add
            "decode": self._new_sparsity_counter(),  #add
        }  #add

        self.unit_sparsity = {  #add
            "full_forward": {},  #add
            "prefill": {},  #add
            "decode": {},  #add
        }  #add

        self.collected_layer_names_by_phase = {  #add
            "full_forward": set(),  #add
            "prefill": set(),  #add
            "decode": set(),  #add
        }  #add

        self.collected_layer_call_count_by_phase = {  #add
            "full_forward": {},  #add
            "prefill": {},  #add
            "decode": {},  #add
        }  #add

        self.current_phase = "full_forward"  #add

    def _accumulate_phase_sparsity(  #add
        self,  #add
        phase: str,  #add
        total_num: int,  #add
        abs_less_th: int,  #add
        total_bits: int,  #add
        zero_bits_total: int,  #add
        sparse_bits_total: int,  #add
        amplitude_zero_bits_total: int,  #add
    ):  #add
        if phase not in self.phase_sparsity:  #add
            self.phase_sparsity[phase] = self._new_sparsity_counter()  #add

        counter = self.phase_sparsity[phase]  #add

        counter["total_zero_count"] += abs_less_th  #add
        counter["total_element_count"] += total_num  #add
        counter["total_bit_count"] += total_bits  #add
        counter["total_0bit_count"] += zero_bits_total  #add
        counter["total_sparsebit_count"] += sparse_bits_total  #add
        counter["total_amplitude_zero_bits_total"] += amplitude_zero_bits_total  #add
        counter["total_amplitude_bit_count"] += total_bits  #add

    def _print_one_sparsity_counter(self, title: str, counter: dict):  #add
        elem = counter["total_element_count"]  #add
        bits = counter["total_bit_count"]  #add

        print(f"\n[{title}]")  #add

        if elem > 0:  #add
            print(  #add
                f"  量化激活零值比例: "  #add
                f"{counter['total_zero_count'] / elem:.4%}"  #add
            )  #add
            print(  #add
                f"  零值元素: "  #add
                f"{counter['total_zero_count']:,} / {elem:,}"  #add
            )  #add
        else:  #add
            print("  未收集到元素级统计")  #add

        if bits > 0:  #add
            print(  #add
                f"  稀疏比特比例: "  #add
                f"{counter['total_sparsebit_count'] / bits:.4%}"  #add
            )  #add
            print(  #add
                f"  所选编码零比特比例: "  #add
                f"{counter['total_amplitude_zero_bits_total'] / bits:.4%}"  #add
            )  #add
            print(  #add
                f"  零比特比例: "  #add
                f"{counter['total_0bit_count'] / bits:.4%}"  #add
            )  #add
        else:  #add
            print("  未收集到bit级统计")  #add

    def print_prefill_decode_sparsity(self):  #add
        print("\n" + "-" * 80)  #add
        print("PREFILL / DECODE SPARSITY")  #add
        print("-" * 80)  #add

        self._print_one_sparsity_counter(  #add
            "PREFILL",  #add
            self.phase_sparsity["prefill"],  #add
        )  #add

        self._print_one_sparsity_counter(  #add
            "DECODE",  #add
            self.phase_sparsity["decode"],  #add
        )  #add

    def print_global_sparsity(self, title: str = "GLOBAL SPARSITY"):  #add
        print("\n" + "-" * 80)  #add
        print(title)  #add
        print("-" * 80)  #add

        counter = {  #add
            "total_zero_count": self.total_zero_count,  #add
            "total_element_count": self.total_element_count,  #add
            "total_bit_count": self.total_bit_count,  #add
            "total_0bit_count": self.total_0bit_count,  #add
            "total_sparsebit_count": self.total_sparsebit_count,  #add
            "total_amplitude_zero_bits_total": self.total_amplitude_zero_bits_total,  #add
            "total_amplitude_bit_count": self.total_amplitude_bit_count,  #add
        }  #add

        self._print_one_sparsity_counter("TOTAL", counter)  #add

    def configure_unit_sparsity(  #add
        self,  #add
        enable: bool = True,  #add
        bit_group_size: int = 2,  #add
        dim_group_size: int = 2,  #add
    ):  #add
        self.enable_unit_sparsity = enable  #add
        self.unit_bit_group_size = int(bit_group_size)  #add
        self.unit_dim_group_size = int(dim_group_size)  #add

        if self.unit_bit_group_size <= 0:  #add
            raise ValueError("unit_bit_group_size must be > 0")  #add

        if self.unit_dim_group_size <= 0:  #add
            raise ValueError("unit_dim_group_size must be > 0")  #add

    def _new_unit_sparsity_counter(self):  #add
        return {  #add
            "zero_units": 0,  #add
            "total_units": 0,  #add
        }  #add

    def _get_unit_counter(self, phase: str, layer_key: str):  #add
        if phase not in self.unit_sparsity:  #add
            self.unit_sparsity[phase] = {}  #add

        if layer_key not in self.unit_sparsity[phase]:  #add
            self.unit_sparsity[phase][layer_key] = self._new_unit_sparsity_counter()  #add

        return self.unit_sparsity[phase][layer_key]  #add

    def record_collected_layer_name(  #add
        self,  #add
        layer_name: str,  #add
        layer_idx: int,  #add
    ):  #add
        phase = getattr(self, "current_phase", "full_forward")  #add
        layer_key = f"{layer_name}_{layer_idx}"  #add

        if not hasattr(self, "collected_layer_names_by_phase"):  #add
            self.collected_layer_names_by_phase = {  #add
                "full_forward": set(),  #add
                "prefill": set(),  #add
                "decode": set(),  #add
            }  #add

        if not hasattr(self, "collected_layer_call_count_by_phase"):  #add
            self.collected_layer_call_count_by_phase = {  #add
                "full_forward": {},  #add
                "prefill": {},  #add
                "decode": {},  #add
            }  #add

        if phase not in self.collected_layer_names_by_phase:  #add
            self.collected_layer_names_by_phase[phase] = set()  #add

        if phase not in self.collected_layer_call_count_by_phase:  #add
            self.collected_layer_call_count_by_phase[phase] = {}  #add

        self.collected_layer_names_by_phase[phase].add(layer_key)  #add

        counter = self.collected_layer_call_count_by_phase[phase]  #add
        counter[layer_key] = counter.get(layer_key, 0) + 1  #add

    def register_layer(self, layer_name: str, layer_idx: int):
        """
        Register a layer for statistics collection.

        Args:
            layer_name: Name of the layer
            layer_idx: Index of the layer
        """
        key = f"{layer_name}_{layer_idx}"
        if key not in self.stats:
            self.stats[key] = QuantStatistics(layer_name, layer_idx)

    def collect_linear_stats(
        self,
        layer_name: str,
        layer_idx: int,
        w_scale: float,
        a_scale: float,
        o_scale: float
    ):
        """Collect statistics for linear layer."""
        key = f"{layer_name}_{layer_idx}"
        if key not in self.stats:
            self.register_layer(layer_name, layer_idx)
        self.stats[key].collect_linear_stats(w_scale, a_scale, o_scale)

    def collect_global_sparsity(
        self,
        total_num: int,
        abs_less_th: int,
        total_bits: int,
        zero_bits_total: int,
        sparse_bits_total: int,
        amplitude_zero_bits_total: int
    ):
        """
        Collect global sparsity statistics.

        Args:
            total_num: Total number of elements in this layer/tensor.
            abs_less_th: Number of elements whose abs value <= threshold.
            total_bits: Total bit count.
            zero_bits_total: Number of zero bits in two's-complement-like representation.
            sparse_bits_total: Sparse-bit count.
            amplitude_zero_bits_total: Number of zero bits in sign-magnitude/original-code representation.
        """
        self.total_zero_count += abs_less_th
        self.total_element_count += total_num
        self.total_bit_count += total_bits
        self.total_0bit_count += zero_bits_total
        self.total_sparsebit_count += sparse_bits_total
        self.total_amplitude_zero_bits_total += amplitude_zero_bits_total
        self.total_amplitude_bit_count += total_bits  #add

    def collect_dynamic_weight_sparsity(
        self,
        total_num: int,
        abs_less_th: int,
        total_bits: int,
        zero_bits_total: int,
        sparse_bits_total: int,
        amplitude_zero_bits_total: int
    ):

        self.dynamic_weight_zero_count += abs_less_th
        self.dynamic_weight_element_count += total_num
        self.dynamic_weight_bit_count += total_bits
        self.dynamic_weight_0bit_count += zero_bits_total
        self.dynamic_weight_sparsebit_count += sparse_bits_total
        self.dynamic_weight_amplitude_zero_bits_total += amplitude_zero_bits_total
        self.dynamic_weight_amplitude_bit_count += total_bits  #add


    def collect_weight_sparsity(
        self,
        total_num: int,
        abs_less_th: int,
        total_bits: int,
        zero_bits_total: int,
        sparse_bits_total: int,
        amplitude_zero_bits_total: int
    ):

        self.weight_zero_count += abs_less_th
        self.weight_element_count += total_num
        self.weight_bit_count += total_bits
        self.weight_0bit_count += zero_bits_total
        self.weight_sparsebit_count += sparse_bits_total
        self.weight_amplitude_zero_bits_total += amplitude_zero_bits_total
        self.weight_amplitude_bit_count += total_bits  #add


    def collect_activation_sparsity(
        self,
        total_num: int,
        abs_less_th: int,
        total_bits: int,
        zero_bits_total: int,
        sparse_bits_total: int,
        amplitude_zero_bits_total: int
    ):
        self.activation_zero_count += abs_less_th
        self.activation_element_count += total_num
        self.activation_bit_count += total_bits
        self.shift_bit_total_count += zero_bits_total
        self.shift_bit_0_count += sparse_bits_total
        self.activation_amplitude_zero_bits_total += amplitude_zero_bits_total
        self.activation_amplitude_bit_count += total_bits  #add

    def collect_matmul_stats(
        self,
        layer_name: str,
        layer_idx: int,
        A_scale: float,
        B_scale: float,
        O_scale: float
    ):
        """Collect statistics for matrix multiplication."""
        key = f"{layer_name}_{layer_idx}"
        if key not in self.stats:
            self.register_layer(layer_name, layer_idx)
        self.stats[key].collect_matmul_stats(A_scale, B_scale, O_scale)

    def export_collected_layers_csv(  #add
        self,  #add
        csv_path: str,  #add
        config_name: str = "",  #add
        model_path: str = "",  #add
        phases=None,  #add
        append: bool = False,  #add
    ):  #add
        import csv  #add
        import os  #add

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)  #add

        if phases is None:  #add
            phases = ["full_forward", "prefill", "decode"]  #add

        file_exists = os.path.exists(csv_path)  #add
        mode = "a" if append else "w"  #add

        with open(csv_path, mode, newline="") as f:  #add
            writer = csv.writer(f)  #add

            if (not file_exists) or (not append):  #add
                writer.writerow([  #add
                    "config",  #add
                    "model_path",  #add
                    "phase",  #add
                    "layer_key",  #add
                    "layer_type",  #add
                    "layer_idx",  #add
                    "call_count",  #add
                ])  #add

            for phase in phases:  #add
                names = sorted(  #add
                    self.collected_layer_names_by_phase.get(phase, set())  #add
                )  #add
                call_counts = self.collected_layer_call_count_by_phase.get(phase, {})  #add

                for layer_key in names:  #add
                    layer_type, layer_idx = self._split_layer_key(layer_key)  #add
                    writer.writerow([  #add
                        config_name,  #add
                        model_path,  #add
                        phase,  #add
                        layer_key,  #add
                        layer_type,  #add
                        layer_idx,  #add
                        call_counts.get(layer_key, 0),  #add
                    ])  #add

    def export_collected_layers_summary_csv(  #add
        self,  #add
        csv_path: str,  #add
        config_name: str = "",  #add
        model_path: str = "",  #add
        phases=None,  #add
        append: bool = True,  #add
    ):  #add
        import csv  #add
        import os  #add

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)  #add

        if phases is None:  #add
            phases = ["full_forward", "prefill", "decode"]  #add

        file_exists = os.path.exists(csv_path)  #add
        mode = "a" if append else "w"  #add

        rows = []  #add

        for phase in phases:  #add
            names = sorted(  #add
                self.collected_layer_names_by_phase.get(phase, set())  #add
            )  #add
            call_counts = self.collected_layer_call_count_by_phase.get(phase, {})  #add

            grouped = {}  #add

            for layer_key in names:  #add
                layer_type, layer_idx = self._split_layer_key(layer_key)  #add

                if layer_type not in grouped:  #add
                    grouped[layer_type] = {  #add
                        "indices": [],  #add
                        "calls": [],  #add
                    }  #add

                if layer_idx >= 0:  #add
                    grouped[layer_type]["indices"].append(layer_idx)  #add

                grouped[layer_type]["calls"].append(  #add
                    int(call_counts.get(layer_key, 0))  #add
                )  #add

            for layer_type in sorted(grouped.keys()):  #add
                indices = sorted(set(grouped[layer_type]["indices"]))  #add
                calls = grouped[layer_type]["calls"]  #add

                if len(indices) > 0:  #add
                    min_idx = min(indices)  #add
                    max_idx = max(indices)  #add
                    expected = set(range(min_idx, max_idx + 1))  #add
                    missing = sorted(expected - set(indices))  #add
                    missing_str = "|".join(str(x) for x in missing)  #add
                else:  #add
                    min_idx = -1  #add
                    max_idx = -1  #add
                    missing_str = ""  #add

                total_calls = sum(calls) if calls else 0  #add
                min_calls = min(calls) if calls else 0  #add
                max_calls = max(calls) if calls else 0  #add

                rows.append([  #add
                    config_name,  #add
                    model_path,  #add
                    phase,  #add
                    layer_type,  #add
                    len(indices),  #add
                    min_idx,  #add
                    max_idx,  #add
                    total_calls,  #add
                    min_calls,  #add
                    max_calls,  #add
                    missing_str,  #add
                ])  #add

        with open(csv_path, mode, newline="") as f:  #add
            writer = csv.writer(f)  #add

            if (not file_exists) or (not append):  #add
                writer.writerow([  #add
                    "config",  #add
                    "model_path",  #add
                    "phase",  #add
                    "layer_type",  #add
                    "num_layers",  #add
                    "min_layer_idx",  #add
                    "max_layer_idx",  #add
                    "total_calls",  #add
                    "min_calls",  #add
                    "max_calls",  #add
                    "missing_layer_indices",  #add
                ])  #add

            writer.writerows(rows)  #add

    def _split_layer_key(self, layer_key: str):  #add
        """  #add
        Parse layer key into layer_type and layer_idx.

        Examples:
            q_proj_0        -> q_proj, 0
            qk_matmul_A_0  -> qk_matmul_A, 0
            pv_matmul_B_23 -> pv_matmul_B, 23
            fc1_10         -> fc1, 10
        """  #add
        return self._split_unit_layer_key(layer_key)  #add

    def Mapping_stat_bf16(
    self,
    x: torch.Tensor,
    in_features: int,
    out_features: int,
    ):
        """
        将单一 BF16 activation 映射到 EffLoc prefill 数据通路。

        输入:
            x:
            BF16 或可转换为 BF16 的 activation。
            支持形状:
                [token, dim]
                [batch, token, dim]
                以及任意最后一维为 in_features 的张量。

        in_features:
            Linear 层输入维度 K。

        out_features:
            Linear 层输出维度 N。

    BF16 统计口径:
        BF16 = 1 sign + 8 exponent + 7 mantissa

        bit-serial workload:
            sign + explicit mantissa = 8 bits

        exponent:
            单独用于计算 bank 内 exponent alignment 开销:
            2 * (max_exp - min_exp)

    返回值单位:
        返回的 latency 都是 effective-bit steps。

        物理时钟数应在外部统一计算:
            physical_cycles =
                effective_bit_steps * cycles_per_effective_bit

        当前硬件:
            cycles_per_effective_bit = 3

        不要在本函数内部再次乘 3。
    """

        h = 64
        w = 48
        nbank = 16
        nmacro = 16

        # 输出通道方向需要多少个 48-column weight tile
        weight_cycles = math.ceil(out_features / w)

        if x is None:
            raise ValueError("Mapping_stat_bf16 requires activation tensor x")

        if x.dim() < 1:
            raise ValueError(
                f"Mapping_stat_bf16 expects at least 1D input, got {x.dim()}D"
            )

        if x.shape[-1] != in_features:
            raise ValueError(
                f"x.shape[-1]={x.shape[-1]}, "
                f"expected in_features={in_features}"
            )

        # ============================================================
        # 1. 将输入统一整理成 [n_tokens, in_features]
        # ============================================================

        x_f32 = torch.nan_to_num(
            x.detach().to(torch.float32),
            nan=0.0,
            posinf=torch.finfo(torch.bfloat16).max,
            neginf=-torch.finfo(torch.bfloat16).max,
        )

        x_2d = x_f32.reshape(-1, in_features)

        n_tokens = x_2d.shape[0]

        # 真正转换为 BF16，确保后续提取的是 BF16 raw bits
        x_bf16 = x_2d.to(torch.bfloat16).contiguous()

        # ============================================================
        # 2. 提取 BF16 raw bit pattern
        #
        # BF16:
        #   bit 15      : sign
        #   bit 14:7    : exponent
        #   bit 6:0     : mantissa
        # ============================================================

        # bfloat16 和 int16 都是 16 bit，可以直接 reinterpret
        raw = (
            x_bf16
            .view(torch.int16)
            .to(torch.int64)
            & 0xFFFF
        )

        sign = (raw >> 15) & 0x1
        exp_codes = (raw >> 7) & 0xFF
        mantissa = raw & 0x7F

        # 非零判断不能只看 exponent，因为 BF16 也可能有 subnormal
        nonzero = (exp_codes != 0) | (mantissa != 0)

        # 0MMM: 只保留 mantissa，不含 sign bit
        # BF16: 0b0MMMMMMM (7 mantissa bits only)
        sm_codes = mantissa

        # +0 和 -0 都不产生有效计算
        sm_codes = torch.where(
            nonzero,
            sm_codes,
            torch.zeros_like(sm_codes),
        )

        exp_codes = torch.where(
            nonzero,
            exp_codes,
            torch.zeros_like(exp_codes),
        )

        # ============================================================
        # 3. K 方向需要多少个 round
        #
        # 一个 macro 每个 K round 能覆盖:
        #   16 banks * 64 dimensions = 1024 dimensions
        # ============================================================

        k_capacity_per_round = nbank * h

        nrowcycle = math.ceil(
            in_features / k_capacity_per_round
        )

        # ============================================================
        # 4. token 数补齐到 16 个 macro 的倍数
        # ============================================================

        pad_tokens = (
            nmacro - n_tokens % nmacro
        ) % nmacro

        padded_tokens = n_tokens + pad_tokens

        ncycle = max(
            1,
            padded_tokens // nmacro,
        )

        sm_padded = F.pad(
            sm_codes,
            (0, 0, 0, pad_tokens),
            value=0,
        )

        exp_padded = F.pad(
            exp_codes,
            (0, 0, 0, pad_tokens),
            value=0,
        )

        # 标记哪些 token 是真实 token，哪些是 padding token
        valid_token_mask = torch.zeros(
            padded_tokens,
            dtype=torch.bool,
            device=x.device,
        )

        valid_token_mask[:n_tokens] = True

        # ============================================================
        # 5. 将 K 均匀分配给所有 K-round × bank
        # ============================================================

        total_bank_slots = nrowcycle * nbank

        base_dims = (
            in_features // total_bank_slots
        )

        extra_slots = (
            in_features % total_bank_slots
        )

        if base_dims > h:
            raise RuntimeError(
                f"Each bank slot requires at least {base_dims} dimensions, "
                f"but bank capacity h={h}"
            )

        valid_dims_per_bank = torch.zeros(
            nrowcycle,
            nbank,
            dtype=torch.long,
            device=x.device,
        )

        # 形状:
        # [token, K-round, bank, dimension-in-bank]
        sm_balanced = torch.zeros(
            padded_tokens,
            nrowcycle,
            nbank,
            h,
            dtype=torch.int64,
            device=x.device,
        )

        exp_balanced = torch.zeros_like(
            sm_balanced
        )

        offset = 0

        for row_cycle in range(nrowcycle):
            for bank_idx in range(nbank):

                slot_idx = (
                    row_cycle * nbank + bank_idx
                )

                valid_dims = (
                    base_dims
                    + (1 if slot_idx < extra_slots else 0)
                )

                if valid_dims > h:
                    raise RuntimeError(
                        f"K-round={row_cycle}, bank={bank_idx} "
                        f"requires {valid_dims} dimensions, but h={h}"
                    )

                valid_dims_per_bank[
                    row_cycle,
                    bank_idx,
                ] = valid_dims

                if valid_dims == 0:
                    continue

                src = slice(
                    offset,
                    offset + valid_dims,
                )

                sm_balanced[
                    :,
                    row_cycle,
                    bank_idx,
                    :valid_dims,
                ] = sm_padded[:, src]

                exp_balanced[
                    :,
                    row_cycle,
                    bank_idx,
                    :valid_dims,
                ] = exp_padded[:, src]

                offset += valid_dims

        if offset != in_features:
            raise RuntimeError(
                f"K mapping error: mapped={offset}, "
                f"expected={in_features}"
            )

        # ============================================================
        # 6. 将 token 分配给 16 个 macro
        #
        # token 0  -> macro 0
        # token 1  -> macro 1
        # ...
        # token 15 -> macro 15
        # token 16 -> macro 0
        # ============================================================

        sm_div = sm_balanced.reshape(
            ncycle,
            nmacro,
            nrowcycle,
            nbank,
            h,
        )

        exp_div = exp_balanced.reshape(
            ncycle,
            nmacro,
            nrowcycle,
            nbank,
            h,
        )

        valid_token_div = valid_token_mask.reshape(
            ncycle,
            nmacro,
        )

        del sm_balanced
        del exp_balanced

        # ============================================================
        # 7. 统计 BF16 sign+mantissa 的有效 1-bit
        #
        # BF16:
        #   1 sign + 7 explicit mantissa = 8 bits
        # ============================================================

        bf16_sm_width = 8

        shifts = torch.arange(
            bf16_sm_width,
            device=x.device,
            dtype=torch.int64,
        )

        # shape:
        # [ncycle, nmacro, nrowcycle, nbank, h, 8]
        bit_values = (
            sm_div.unsqueeze(-1) >> shifts
        ) & 1

        # 对 h=64 个 dimension 和 8 个 S+M bit 求和
        popcount_sum = bit_values.sum(
            dim=(-1, -2)
        ).float()

        # shape:
        # [ncycle, nmacro, nrowcycle, nbank]

        # ============================================================
        # 8. 计算每个 bank 的 exponent range
        # ============================================================

        nonzero_mask = (
            (exp_div != 0)
            | (sm_div != 0)
        )

        large = torch.full_like(
            exp_div,
            1 << 30,
        )

        small = torch.full_like(
            exp_div,
            -(1 << 30),
        )

        exp_min = torch.where(
            nonzero_mask,
            exp_div,
            large,
        ).min(dim=-1).values

        exp_max = torch.where(
            nonzero_mask,
            exp_div,
            small,
        ).max(dim=-1).values

        all_zero_bank = ~nonzero_mask.any(
            dim=-1
        )

        exp_min = torch.where(
            all_zero_bank,
            torch.zeros_like(exp_min),
            exp_min,
        )

        exp_max = torch.where(
            all_zero_bank,
            torch.zeros_like(exp_max),
            exp_max,
        )

        exp_range = (
            exp_max - exp_min
        ).float()

        # ============================================================
        # 9. 每个 bank 的实际工作量
        #
        # 与原 FP8 Mapping_stat 保持相同模型:
        #
        # bank_steps =
        #     effective sign/mantissa 1-bits
        #     + 2 * exponent range
        # ============================================================

        bank_steps = (
            popcount_sum
            + 2.0 * exp_range
        )

        # padding token 强制为 0 workload
        bank_steps = (
            bank_steps
            * valid_token_div[
                :,
                :,
                None,
                None,
            ].float()
        )

        # ============================================================
        # 10. Macro 内部 bank barrier
        #
        # 16 个 bank 并行，但一个 K-round 的延迟由最慢 bank 决定
        # ============================================================

        round_latency = bank_steps.amax(
            dim=-1
        )

        # shape:
        # [ncycle, nmacro, nrowcycle]

        # 每个 macro 顺序处理:
        #   自己的所有 token
        #   每个 token 的所有 K-round
        macro_total = round_latency.sum(
            dim=(0, 2)
        )

        # shape: [nmacro]

        # 16 个 macro 异步执行，整层由最慢 macro 决定
        layer_latency_single_weight_tile = (
            macro_total.max()
        )

        # ============================================================
        # 11. BF16 dense baseline
        #
        # 每个真实 BF16 activation:
        #   8 个 S+M bit 全部视为有效
        # ============================================================

        dense_per_bank = (
            valid_token_div[
                :,
                :,
                None,
                None,
            ].float()
            * valid_dims_per_bank[
                None,
                None,
                :,
                :,
            ].float()
            * bf16_sm_width
        )

        # shape:
        # [ncycle, nmacro, nrowcycle, nbank]

        dense_round_latency = dense_per_bank.amax(
            dim=-1
        )

        dense_macro_total = dense_round_latency.sum(
            dim=(0, 2)
        )

        bf16_sm_width = 8

        dense_steps_per_bank = (
            valid_dims_per_bank.float()
            * bf16_sm_width
        )

        # 一个 K-round 的延迟由 16 个 bank 中负载最大的决定
        dense_steps_per_round = (
            dense_steps_per_bank.amax(dim=-1)
        )

        # 所有 token cycle 和所有 K-round
        layer_ideal_single_weight_tile = (
            dense_steps_per_round.sum()
            * ncycle
        )

        layer_ideal_latency = (
            layer_ideal_single_weight_tile
            * weight_cycles
        )

        # ============================================================
        # 12. 乘输出通道方向的 weight tile 数
        # ============================================================

        layer_latency = (
            layer_latency_single_weight_tile
            * weight_cycles
        )

        layer_ideal_latency = (
            layer_ideal_single_weight_tile
            * weight_cycles
        )

        # 所有输出 weight tile 对应的 per-bank workload
        effective_bank_steps_all_tiles = (
            bank_steps * weight_cycles
        )

        return (
            bank_steps,
            layer_latency_single_weight_tile,
            layer_ideal_single_weight_tile,
            layer_latency,
            layer_ideal_latency,
            effective_bank_steps_all_tiles,
        )

    def _extract_sm_codes_from_mp(
        self,
        mp_codes: list,
        n_tokens: int,
        in_features: int,
        device,
    ):
        """Scatter local mixed-precision codes back into global token order.

        The hardware code follows the 0MMM convention: mantissa bits only,
        no sign bit (e.g. 0MMM for E4M3, 0MMMMMMM for BF16). The hidden one
        is not inserted because the ordinary mapping path does not count it.
        """
        sm_codes = torch.zeros(
            n_tokens, in_features, dtype=torch.int64, device=device
        )
        sm_widths = torch.zeros(n_tokens, dtype=torch.int64, device=device)
        exp_codes = torch.zeros(
            n_tokens, in_features, dtype=torch.int64, device=device
        )

        covered = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        for code_local, indices, spec in mp_codes:
            if indices.numel() == 0:
                continue
            if code_local.dim() != 2 or code_local.shape[1] != in_features:
                raise ValueError(
                    f"mixed-precision local code must be [N, {in_features}], "
                    f"got {tuple(code_local.shape)}"
                )
            if code_local.shape[0] != indices.numel():
                raise ValueError(
                    "local code rows and global index count do not match: "
                    f"{code_local.shape[0]} vs {indices.numel()}"
                )

            fmt = (getattr(spec, "fmt", "") or "").lower().strip()
            sm_local, width, exp_local = self._encode_fp_values_to_sm_exp(
                code_local, fmt
            )
            sm_codes.index_copy_(0, indices, sm_local)
            exp_codes.index_copy_(0, indices, exp_local)
            sm_widths.index_fill_(0, indices, int(width))
            covered.index_fill_(0, indices, True)

        if n_tokens > 0 and not bool(covered.all()):
            missing = torch.nonzero(~covered, as_tuple=False).flatten()
            raise RuntimeError(
                f"mixed-precision codes do not cover all tokens; "
                f"missing {missing.numel()} token(s)"
            )
        return sm_codes, sm_widths, exp_codes

    @staticmethod
    def _unpack_sm_exp(
        raw: torch.Tensor,
        sign_shift: int,
        exp_bits: int,
        mant_bits: int,
    ):
        """Unpack exponent and explicit mantissa from a raw FP code (0MMM format).

        Only mantissa bits are kept in sm; the sign bit is excluded.
        This matches the 0MMM convention used in 0608_macro_layout_mapping.py:
          FP8 E4M3:  0b0MMM  (3 mantissa bits)
          BF16 E8M7: 0b0MMMMMMM (7 mantissa bits)
          FP16 E5M10: 0b0MMMMMMMMM (10 mantissa bits)
          FP4 E2M1:  0b0M (1 mantissa bit)
        """
        raw = raw.to(torch.int64)
        mant = raw & ((1 << mant_bits) - 1)
        exp = (raw >> mant_bits) & ((1 << exp_bits) - 1)
        sm = mant  # 0MMM: mantissa only, no sign bit
        nonzero = (exp != 0) | (mant != 0)
        return sm, exp, nonzero

    def _extract_sm_from_raw(
        self,
        raw_int: torch.Tensor,
        fmt: str,
    ):
        """Extract mantissa-only codes (0MMM format), matching ordinary Mapping_stat."""
        fmt = (fmt or "").lower().strip()
        if fmt in {"e5m10", "fp16", "float16"}:
            sm, exp, _ = self._unpack_sm_exp(
                raw_int, sign_shift=15, exp_bits=5, mant_bits=10
            )
            return sm, 10, exp  # 0MMM: 10 mantissa bits only (no sign)
        if fmt in {"e4m3", "fp8_e4m3", "float8_e4m3"}:
            sm, exp, _ = self._unpack_sm_exp(
                raw_int, sign_shift=7, exp_bits=4, mant_bits=3
            )
            return sm, 3, exp  # 0MMM: 3 mantissa bits only (no sign)
        if fmt in {"e2m1", "fp4", "fp4_e2m1"}:
            sm, exp, _ = self._unpack_sm_exp(
                raw_int, sign_shift=3, exp_bits=2, mant_bits=1
            )
            return sm, 1, exp  # 0MMM: 1 mantissa bit only (no sign)
        if fmt in {"bf16", "bfloat16"}:
            sm, exp, _ = self._unpack_sm_exp(
                raw_int, sign_shift=15, exp_bits=8, mant_bits=7
            )
            return sm, 7, exp  # 0MMM: 7 mantissa bits only (no sign)
        raise ValueError(f"Unsupported fmt in _extract_sm_from_raw: {fmt}")

    def _encode_fp_values_to_sm_exp(
        self,
        values: torch.Tensor,
        fmt: str,
    ):
        """Convert quantized numerical values to exact format codes for statistics."""
        fmt = (fmt or "").lower().strip()
        values_f32 = torch.nan_to_num(
            values.detach().to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

        def _canonicalize_zero(result):
            sm, width, exp = result
            zero = values_f32 == 0
            sm = torch.where(zero, torch.zeros_like(sm), sm)
            exp = torch.where(zero, torch.zeros_like(exp), exp)
            return sm, width, exp

        if fmt in {"e5m10", "fp16", "float16"}:
            raw = values_f32.to(torch.float16).view(torch.int16).to(torch.int64)
            return _canonicalize_zero(self._extract_sm_from_raw(raw, "e5m10"))

        if fmt in {"e4m3", "fp8_e4m3", "float8_e4m3"}:
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError("this PyTorch build does not support float8_e4m3fn")
            raw = values_f32.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int64)
            return _canonicalize_zero(self._extract_sm_from_raw(raw, "e4m3"))

        if fmt in {"e2m1", "fp4", "fp4_e2m1"}:
            # E2M1 magnitude codes: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
            abs_v = values_f32.abs()
            mag_code = torch.zeros_like(abs_v, dtype=torch.int64)
            mag_code[abs_v >= 5.0] = 7
            mag_code[(abs_v >= 3.5) & (abs_v < 5.0)] = 6
            mag_code[(abs_v >= 2.5) & (abs_v < 3.5)] = 5
            mag_code[(abs_v >= 1.75) & (abs_v < 2.5)] = 4
            mag_code[(abs_v >= 1.25) & (abs_v < 1.75)] = 3
            mag_code[(abs_v >= 0.75) & (abs_v < 1.25)] = 2
            mag_code[(abs_v >= 0.25) & (abs_v < 0.75)] = 1
            sign = (values_f32 < 0).to(torch.int64)
            raw = (sign << 3) | mag_code
            return _canonicalize_zero(self._extract_sm_from_raw(raw, "e2m1"))

        if fmt in {"bf16", "bfloat16"}:
            raw = values_f32.to(torch.bfloat16).view(torch.int16).to(torch.int64)
            return _canonicalize_zero(self._extract_sm_from_raw(raw, "bf16"))

        raise ValueError(f"Unsupported floating-point format: {fmt}")

    def _extract_exponent_from_scalar(
        self,
        value: float,
        fmt: str,
    ) -> int:
        """Extract the actual exponent (bias subtracted) from a Python float in the given FP format.

        Converts *value* to the target format (E4M3, BF16, FP16, E2M1),
        reinterprets the raw bit pattern, and returns the **unbiased** exponent.

        For normal values:  actual_exp = stored_exp - bias
        For subnormals:     actual_exp = 1 - bias  (by IEEE convention)
        For zero:           returns 0

        Bias values:
            E4M3  bias = 7
            E2M1  bias = 1
            E5M10 bias = 15
            BF16  bias = 127
            FP32  bias = 127
        """
        import struct

        if value == 0.0:
            return 0

        fmt = (fmt or "").lower().strip()
        abs_val = abs(value)

        if fmt in {"e4m3", "fp8_e4m3", "float8_e4m3"}:
            # FP8 E4M3: 1 sign + 4 exp + 3 mant, bias = 7
            bias = 7
            raw = torch.tensor([abs_val], dtype=torch.float32).to(
                torch.float8_e4m3fn
            ).view(torch.uint8).item()
            stored_exp = (raw >> 3) & 0xF  # bits [6:3]
            if stored_exp == 0:
                # subnormal: actual_exp = 1 - bias
                return 1 - bias
            return int(stored_exp) - bias

        if fmt in {"e2m1", "fp4", "fp4_e2m1"}:
            # FP4 E2M1: 1 sign + 2 exp + 1 mant, bias = 1
            #   stored_exp | actual_exp | magnitude range
            #       0      |  1-1 = 0   | 0, 0.5  (subnormal)
            #       1      |  1-1 = 0   | 1, 1.5
            #       2      |  2-1 = 1   | 2, 3
            #       3      |  3-1 = 2   | 4, 6
            bias = 1
            if abs_val >= 4.0:
                stored_exp = 3
            elif abs_val >= 2.0:
                stored_exp = 2
            elif abs_val >= 1.0:
                stored_exp = 1
            elif abs_val > 0.0:
                stored_exp = 0  # subnormal
            else:
                return 0
            if stored_exp == 0:
                return 1 - bias
            return int(stored_exp) - bias

        if fmt in {"e5m10", "fp16", "float16"}:
            # FP16 E5M10: 1 sign + 5 exp + 10 mant, bias = 15
            bias = 15
            raw = struct.pack('>e', abs_val)  # half-precision, big-endian
            raw_int = struct.unpack('>H', raw)[0]
            stored_exp = (raw_int >> 10) & 0x1F  # bits [14:10]
            if stored_exp == 0:
                return 1 - bias
            return int(stored_exp) - bias

        if fmt in {"bf16", "bfloat16"}:
            # BF16 E8M7: 1 sign + 8 exp + 7 mant, bias = 127
            bias = 127
            raw = torch.tensor([abs_val], dtype=torch.float32).view(torch.int32).item()
            # BF16 = upper 16 bits of FP32
            bf16_bits = (raw >> 16) & 0xFFFF
            stored_exp = (bf16_bits >> 7) & 0xFF  # bits [14:7]
            if stored_exp == 0:
                return 1 - bias
            return int(stored_exp) - bias

        if fmt in {"fp32", "float32"}:
            # FP32 E8M23: 1 sign + 8 exp + 23 mant, bias = 127
            bias = 127
            raw = struct.pack('>f', abs_val)
            raw_int = struct.unpack('>I', raw)[0]
            stored_exp = (raw_int >> 23) & 0xFF  # bits [30:23]
            if stored_exp == 0:
                return 1 - bias
            return int(stored_exp) - bias
            return int(exp)

        raise ValueError(f"Unsupported FP format for exponent extraction: {fmt}")



    def Save_mapping_stat(self, path: str):
        """
        Save collected mapping statistics to a directory using pickle.

        Args:
            path: Target directory where mapping_stat will be saved.
        """

        def _to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.cpu()
            if isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_cpu(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_cpu(v) for v in obj)
            return obj

        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, "self.mapping_stat_bf16_stat_b2.p")
        mapping_stat_to_save = _to_cpu(self.mapping_stat_bf16_stat_b2)
        with open(filename, 'wb') as f:
            pickle.dump(mapping_stat_to_save, f)

        filename = os.path.join(path, "self.mapping_stat_fp8_stat_b2.p")
        mapping_stat_to_save = _to_cpu(self.mapping_stat_fp8_stat_b2)
        with open(filename, 'wb') as f:
            pickle.dump(mapping_stat_to_save, f)

        filename = os.path.join(path, "self.mapping_stat_int8_stat_b2.p")
        mapping_stat_to_save = _to_cpu(self.mapping_stat_int8_stat_b2)
        with open(filename, 'wb') as f:
            pickle.dump(mapping_stat_to_save, f)

        filename = os.path.join(path, "self.mapping_stat_bf16_stat_b1.p")
        mapping_stat_to_save = _to_cpu(self.mapping_stat_bf16_stat_b1)
        with open(filename, 'wb') as f:
            pickle.dump(mapping_stat_to_save, f)

        filename = os.path.join(path, "self.mapping_stat_fp8_stat_b1.p")
        mapping_stat_to_save = _to_cpu(self.mapping_stat_fp8_stat_b1)
        with open(filename, 'wb') as f:
            pickle.dump(mapping_stat_to_save, f)

        filename = os.path.join(path, "self.mapping_stat_int8_stat_b1.p")
        mapping_stat_to_save = _to_cpu(self.mapping_stat_int8_stat_b1)
        with open(filename, 'wb') as f:
            pickle.dump(mapping_stat_to_save, f)

    def _ensure_latency_log_path(self) -> str:
        """返回 latency_log.jsonl 的完整路径，必要时创建目录。"""
        path = os.path.join(self.scale_dir, "latency_log.jsonl")
        return path

    def append_latency_record(
        self,
        layer_name: str,
        layer_idx: int,
        effective_cim: float,
        base_sum: float,
        utilization_ratio: float,
        sacim_latency_increment: float,
        all_boperation: float,
    ):
        """
        把本次 collect_quant_activation 中的延迟统计数据
        以 JSON Lines 格式追加写入 latency_log.jsonl 的最后一行。
        每次调用都会 open→写入一行→close，避免内存堆积。
        """
        import json
        import os
        import time as _time

        filepath = "/latency_log.jsonl"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        record = {
            "timestamp": _time.time(),
            "phase": getattr(self, "current_phase", "unknown"),
            "nmacro": getattr(self, "nmacro", None),
            "layer_name": layer_name,
            "layer_idx": layer_idx,
            "effective_cim": float(effective_cim),
            "base_sum": float(base_sum),
            "utilization_ratio": float(utilization_ratio),
            "sacim_latency_increment": float(sacim_latency_increment),
            "all_boperation": float(all_boperation),
        }

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _cycles_to_seconds(self, effective_bit_steps) -> float:
        """Convert effective-bit steps to seconds using the hardware timing model."""
        if torch.is_tensor(effective_bit_steps):
            effective_bit_steps = effective_bit_steps.detach().item()
        physical_cycles = (
            float(effective_bit_steps) * float(self.cycles_per_effective_bit)
        )
        return physical_cycles / float(self.clock_freq_hz)

    def _record_per_layer_latency(
        self,
        layer_name: str,
        layer_idx: int,
        layer_latency,
        layer_ideal_latency,
        effective_cim,
        base_sum,
        boperation,
        ideal_sparsity_op=None,
    ):
        """Record per-layer prefill latency using three clocks per effective bit."""
        key = f"{layer_name}_{layer_idx}"
        if key not in self.per_layer_latency:
            self.per_layer_latency[key] = {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "SACIM_latency": 0.0,
                "baseline_latency": 0.0,
                "ideal_sparsity_latency": 0.0,
            }

        entry = self.per_layer_latency[key]
        entry.setdefault("SACIM_latency", 0.0)
        entry.setdefault("baseline_latency", 0.0)
        entry.setdefault("ideal_sparsity_latency", 0.0)
        entry["SACIM_latency"] += self._cycles_to_seconds(layer_latency)
        entry["baseline_latency"] += self._cycles_to_seconds(layer_ideal_latency)

        if ideal_sparsity_op is not None:
            # 16 macros x 16 banks = 256 bank lanes.
            ideal_cycles = float(ideal_sparsity_op.sum()) / 256.0
            entry["ideal_sparsity_latency"] += self._cycles_to_seconds(ideal_cycles)

        baseline = entry["baseline_latency"]
        entry["actual_ideal_speedup"] = (
            baseline / entry["ideal_sparsity_latency"]
            if entry["ideal_sparsity_latency"] > 0
            else 0.0
        )
        entry["actual_actul_speedup"] = (
            baseline / entry["SACIM_latency"]
            if entry["SACIM_latency"] > 0
            else 0.0
        )
        entry["CIM_utilization_ratio"] = (
            entry["SACIM_latency"] / baseline if baseline > 0 else 0.0
        )
        entry["actual_total_bits"] = float(boperation) * 16 * 16
        entry["actual_1_bits"] = float(base_sum) * 16 * 16
        entry["ideal_1_bits"] = float(effective_cim)

    def collect_quant_activation_mixed_precision(
        self,
        layer_name: str,
        layer_idx: int,
        activation_combined: torch.Tensor,
        mp_codes: list,
        weight: torch.Tensor,
        w_spec: QuantSpec,
        digit_size: int,
        parallelism: int,
        in_features: int,
        out_features: int,
        n_heads: int = 1,
        mp_high_ratio: float = 0.1,
        mp_low_ratio: float = 0.65,
    ):
        """
        mixed_precision 专用统计。

        activation_combined: 拼回 [N, H] 的全激活，每个 token 按其档位量化。
                             主要用于 Mapping_stat（一次性算延迟，无屏障）。
        mp_codes: [(code_2d, indices, spec), ...] 共三档。
        n_heads: attention matmul (qk/pv) 时激活行数 = num_heads × S,
                 由调用方从原始张量形状传入 (如 Qwen2.5-14B=40);
                 线性层固定 1, 不参与 reshape。

        设计要点：
        1. sparse_stats 按每档的 spec 分别计算稀疏度，然后求和
           （不同档位宽不同，必须分开统计）
        2. Mapping_stat 直接用 activation_combined 一次性调用，
           对所有 token 做 macro 异步调度，不加跨档屏障同步
        """
        if activation_combined is None:
            return

        self.record_collected_layer_name(layer_name, layer_idx)

        Nmacro = getattr(self, 'nmacro', 16)
        CIM = CIM_sys(
            h=self.cim_geometry["height"],
            w=self.cim_geometry["width"],
            Nadder=self.cim_geometry["banks"],
            Nmacro=Nmacro,
            freq=int(1e9),
        )
        as_l = getattr(self, 'as_l', 1)
        is_prefill = (getattr(self, "current_phase", "") == "prefill")
        Nbankall = CIM.Nmacro * CIM.Nadder

        # ============================================================
        # 1. 构造 per-token sm_codes 和 sm_widths
        #    从 mp_codes 提取各精度 sign+mantissa 整数码
        # ============================================================
        n_tokens = activation_combined.shape[0]
        sm_codes, sm_widths, exp_codes = self._extract_sm_codes_from_mp(
            mp_codes, n_tokens, in_features,
            activation_combined.device,
        )
        # ============================================================
        # 2. Mapping_stat_dynamic: per-token 混精延迟统计
        # ============================================================
        if(layer_name == "q_proj" or layer_name == "k_proj" or layer_name == "v_proj"):
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat_dynamic(CIM,sm_codes, sm_widths, exp_codes, in_features, out_features, False, mp_high_ratio=mp_high_ratio, mp_low_ratio=mp_low_ratio, is_prefill=is_prefill, as_l=self.as_l)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.q_proj_sparsity_speedup = self.q_proj_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.q_proj_SACIM_latency_stat = self.q_proj_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.q_proj_baseline_latency_stat = self.q_proj_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
            else:
                pass

        elif(layer_name == "qk_matmul"):
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat_dynamic(CIM,sm_codes, sm_widths, exp_codes, in_features, out_features, True, n_heads=n_heads, mp_high_ratio=mp_high_ratio, mp_low_ratio=mp_low_ratio, is_prefill=is_prefill, as_l=self.as_l)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.qk_matmul_sparsity_speedup = self.qk_matmul_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.qk_matmul_SACIM_latency_stat = self.qk_matmul_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.qk_matmul_baseline_latency_stat = self.qk_matmul_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
                self._record_per_layer_latency(layer_name, layer_idx, layer_latency, layer_ideal_latency, effective_cim, base, boperation, ideal_sparsity_op)
            else:
                pass
        elif(layer_name == "pv_matmul"):
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat_dynamic(CIM,sm_codes, sm_widths, exp_codes, in_features, out_features, True, n_heads=n_heads, mp_high_ratio=mp_high_ratio, mp_low_ratio=mp_low_ratio, is_prefill=is_prefill, as_l=self.as_l)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.pv_matmul_sparsity_speedup = self.pv_matmul_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.pv_matmul_SACIM_latency_stat = self.pv_matmul_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.pv_matmul_baseline_latency_stat = self.pv_matmul_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
                self._record_per_layer_latency(layer_name, layer_idx, layer_latency, layer_ideal_latency, effective_cim, base, boperation, ideal_sparsity_op)
            else:
                pass

        elif(layer_name == "o_proj" or layer_name == "out_proj"):
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat_dynamic(CIM,sm_codes, sm_widths, exp_codes, in_features, out_features, False, mp_high_ratio=mp_high_ratio, mp_low_ratio=mp_low_ratio, is_prefill=is_prefill, as_l=self.as_l)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.o_proj_sparsity_speedup = self.o_proj_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.o_proj_SACIM_latency_stat = self.o_proj_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.o_proj_baseline_latency_stat = self.o_proj_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
            else:
                pass
        elif(layer_name == "gate_proj"):
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat_dynamic(CIM,sm_codes, sm_widths, exp_codes, in_features, out_features, False, mp_high_ratio=mp_high_ratio, mp_low_ratio=mp_low_ratio, is_prefill=is_prefill, as_l=self.as_l)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.gate_proj_sparsity_speedup = self.gate_proj_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.gate_proj_SACIM_latency_stat = self.gate_proj_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.gate_proj_baseline_latency_stat = self.gate_proj_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
            else:
                pass
        elif(layer_name == "up_proj" or layer_name == "fc1"):
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat_dynamic(CIM,sm_codes, sm_widths, exp_codes, in_features, out_features, False, mp_high_ratio=mp_high_ratio, mp_low_ratio=mp_low_ratio, is_prefill=is_prefill, as_l=self.as_l)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.up_proj_sparsity_speedup = self.up_proj_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.up_proj_SACIM_latency_stat = self.up_proj_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.up_proj_baseline_latency_stat = self.up_proj_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
        elif(layer_name == "down_proj" or layer_name == "fc2"):
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat_dynamic(CIM,sm_codes, sm_widths, exp_codes, in_features, out_features, False, mp_high_ratio=mp_high_ratio, mp_low_ratio=mp_low_ratio, is_prefill=is_prefill, as_l=self.as_l)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.down_proj_sparsity_speedup = self.down_proj_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.down_proj_SACIM_latency_stat = self.down_proj_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.down_proj_baseline_latency_stat = self.down_proj_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
        # # ============================================================
        # # 2. sparse stats: 按每档分别计算，然后求和
        # # ============================================================
        # total_num = 0
        # abs_less_th = 0
        # total_bits = 0
        # zero_bits_total = 0
        # sparse_bits_total = 0
        # amplitude_zero_bits_total = 0

        # for code, indices, spec in mp_codes:
        #     if indices.numel() == 0:
        #         continue

        #     if spec is None or spec.kind in {"none"}:
        #         continue

        #     if spec.kind == "int":
        #         sparse = self.compute_sparse_stats(
        #             spec.bits,
        #             code,
        #             0,
        #             chunk_size=self.sparse_stat_chunk_size,
        #         )
        #     elif spec.kind in {"fp", "bf"}:
        #         sparse = self.compute_sparse_stats_fp(
        #             spec.fmt,
        #             code,
        #             0.0,
        #             chunk_size=self.sparse_stat_chunk_size,
        #         )
        #     else:
        #         continue

        #     (
        #         _num, _less, _bits, _zero, _sparse, _amp
        #     ) = sparse

        #     total_num += _num
        #     abs_less_th += _less
        #     total_bits += _bits
        #     zero_bits_total += _zero
        #     sparse_bits_total += _sparse
        #     amplitude_zero_bits_total += _amp

        # if w_spec.kind == "int":  #add
        #     wt_sparse = self.compute_sparse_stats(  #add
        #         w_spec.bits,  #add
        #         weight,  #add
        #         0,  #add
        #         chunk_size=self.sparse_stat_chunk_size,  #add
        #     )  #add
        # elif w_spec.kind == "fp":  #add
        #     wt_sparse = self.compute_sparse_stats_fp(  #add
        #         w_spec.fmt,  #add
        #         weight,  #add
        #         0.0,  #add
        #         chunk_size=self.sparse_stat_chunk_size,  #add
        #     )  #add
        # elif w_spec.kind == "bf":  #add
        #     wt_sparse = self.compute_sparse_stats_fp(  #add
        #         w_spec.fmt,  #add
        #         weight,  #add
        #         0.0,  #add
        #         chunk_size=self.sparse_stat_chunk_size,  #add
        #     )  #add
        # else:  #add
        #     pass


        # if(layer_name == "qk_matmul" or layer_name == "pv_matmul"):
        #     self.collect_dynamic_weight_sparsity(*wt_sparse)
        # else:
        #     self.collect_weight_sparsity(*wt_sparse)

        # # 累加到全局
        # self.collect_activation_sparsity(
        #     total_num,
        #     abs_less_th,
        #     total_bits,
        #     zero_bits_total,
        #     sparse_bits_total,
        #     amplitude_zero_bits_total,
        # )

        # # 累加到 per-layer
        # _key = f"{layer_name}_{layer_idx}"
        # if _key not in self.per_layer_latency:
        #     self.per_layer_latency[_key] = {
        #         "layer_name": layer_name,
        #         "layer_idx": layer_idx,
        #         "SACIM_latency": 0.0,
        #         "baseline_latency": 0.0,
        #         "effective_cim": 0.0,
        #         "base_sum": 0.0,
        #     }
        # _entry = self.per_layer_latency[_key]
        # _entry["total_elements"] = _entry.get("total_elements", 0) + int(total_num)
        # _entry["zero_elements"] = _entry.get("zero_elements", 0) + int(abs_less_th)
        # _entry["total_bits"] = _entry.get("total_bits", 0) + int(total_bits)
        # _entry["zero_bits"] = _entry.get("zero_bits", 0) + int(zero_bits_total)
        # _entry["sparse_bits"] = _entry.get("sparse_bits", 0) + int(sparse_bits_total)
        # _entry["amplitude_zero_bits"] = (
        #     _entry.get("amplitude_zero_bits", 0) + int(amplitude_zero_bits_total)
        # )
        # _entry["zero_rate"] = (
        #     _entry["zero_elements"] / _entry["total_elements"]
        #     if _entry["total_elements"] > 0
        #     else 0.0
        # )
        # _entry["sparse_bit_rate"] = (
        #     _entry["sparse_bits"] / _entry["total_bits"]
        #     if _entry["total_bits"] > 0
        #     else 0.0
        # )
        # _entry["amplitude_zero_bit_rate"] = (
        #     _entry["amplitude_zero_bits"] / _entry["total_bits"]
        #     if _entry["total_bits"] > 0
        #     else 0.0
        # )
        # _entry["ideal_speed_up"] = (
        #     1 / (1 - _entry["sparse_bit_rate"])
        #     if _entry["sparse_bit_rate"] < 1
        #     else float("inf")
        # )
        # _entry["1_bits"] = (
        #     _entry["total_bits"] - _entry["amplitude_zero_bits"]
        # )

        # # unit_sparsity 也分档统计求和
        # for code, indices, spec in mp_codes:
        #     if indices.numel() == 0:
        #         continue
        #     if spec is None or spec.kind in {"none"}:
        #         continue
        #     self.collect_unit_sparsity(layer_name, layer_idx, code, spec)

    def collect_latency_prefill(
        self,
        CIM: CIM_sys,
        layer_name: str,
        layer_idx: int,
        FP_activation: torch.Tensor,
        in_features: int,
        out_features: int,
        method: str = "default"
    ):
        is_prefill = True
        Nbankall = CIM.Nmacro * CIM.Nadder
        if layer_name in {"q_proj", "k_proj", "v_proj"}:
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat(CIM, FP_activation, in_features, out_features, is_prefill, method)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.q_proj_sparsity_speedup = self.q_proj_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.q_proj_SACIM_latency_stat = self.q_proj_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.q_proj_baseline_latency_stat = self.q_proj_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
                self._record_per_layer_latency(layer_name, layer_idx, layer_latency, layer_ideal_latency, effective_cim, base, boperation, ideal_sparsity_op)
            else:
                pass
        elif(layer_name == "qk_matmul"):
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat(CIM, FP_activation, in_features, out_features, is_prefill, method)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.qk_matmul_sparsity_speedup = self.qk_matmul_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.qk_matmul_SACIM_latency_stat = self.qk_matmul_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.qk_matmul_baseline_latency_stat = self.qk_matmul_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
                self._record_per_layer_latency(layer_name, layer_idx, layer_latency, layer_ideal_latency, effective_cim, base, boperation, ideal_sparsity_op)
            else:
                pass
        elif(layer_name == "pv_matmul"):
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat(CIM, FP_activation, in_features, out_features, is_prefill, method)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.pv_matmul_sparsity_speedup = self.pv_matmul_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.pv_matmul_SACIM_latency_stat = self.pv_matmul_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.pv_matmul_baseline_latency_stat = self.pv_matmul_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
                self._record_per_layer_latency(layer_name, layer_idx, layer_latency, layer_ideal_latency, effective_cim, base, boperation, ideal_sparsity_op)
            else:
                pass
        elif layer_name in {"o_proj", "out_proj"}:
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat(CIM, FP_activation, in_features, out_features, is_prefill, method)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.o_proj_sparsity_speedup = self.o_proj_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.o_proj_SACIM_latency_stat = self.o_proj_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.o_proj_baseline_latency_stat = self.o_proj_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
                self._record_per_layer_latency(layer_name, layer_idx, layer_latency, layer_ideal_latency, effective_cim, base, boperation, ideal_sparsity_op)
            else:
                pass
        elif(layer_name == "gate_proj"):
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat(CIM, FP_activation, in_features, out_features, is_prefill, method)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.gate_proj_sparsity_speedup = self.gate_proj_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.gate_proj_SACIM_latency_stat = self.gate_proj_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.gate_proj_baseline_latency_stat = self.gate_proj_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
                self._record_per_layer_latency(layer_name, layer_idx, layer_latency, layer_ideal_latency, effective_cim, base, boperation, ideal_sparsity_op)
            else:
                pass
        elif layer_name in {"up_proj", "fc1"}:
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat(CIM, FP_activation, in_features, out_features, is_prefill, method)
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.up_proj_sparsity_speedup = self.up_proj_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.up_proj_SACIM_latency_stat = self.up_proj_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.up_proj_baseline_latency_stat = self.up_proj_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
                self._record_per_layer_latency(layer_name, layer_idx, layer_latency, layer_ideal_latency, effective_cim, base, boperation, ideal_sparsity_op)
        elif layer_name in {"down_proj", "fc2"}:
            cim, base, boperation, layer_latency, layer_ideal_latency, ideal_sparsity_op = Mapping_stat(CIM, FP_activation, in_features, out_features, is_prefill, method)
            effective_cim = cim.sum()
            effective_cim = cim.sum()
            all_boperation = boperation
            utlization_ratio = effective_cim / (base.sum() * CIM.Nmacro * CIM.Nadder)
            if True:
                self.down_proj_sparsity_speedup = self.down_proj_sparsity_speedup + (ideal_sparsity_op.sum())/CIM.freq
                self.down_proj_SACIM_latency_stat = self.down_proj_SACIM_latency_stat + self._cycles_to_seconds(layer_latency)
                self.down_proj_baseline_latency_stat = self.down_proj_baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)

                
                self.SACIM_latency_stat = self.SACIM_latency_stat + utlization_ratio * all_boperation * CIM.Nmacro * CIM.Nadder
                self.ideal_sparsity_latency_stat = self.ideal_sparsity_latency_stat + (ideal_sparsity_op.sum())/CIM.freq

                self.latency = self.latency + (layer_latency)/CIM.freq
                self.baseline_latency_stat = self.baseline_latency_stat + self._cycles_to_seconds(layer_ideal_latency)
                self._record_per_layer_latency(layer_name, layer_idx, layer_latency, layer_ideal_latency, effective_cim, base, boperation, ideal_sparsity_op)
                self.append_latency_record(
                    layer_name=layer_name,
                    layer_idx=layer_idx,
                    effective_cim=effective_cim,
                    base_sum=base.sum(),
                    utilization_ratio=utlization_ratio,
                    sacim_latency_increment=utlization_ratio * all_boperation,
                    all_boperation=all_boperation,
                )
            # self._record_per_layer_latency(layer_name, layer_idx, layer_latency, layer_ideal_latency, effective_cim, base, boperation, ideal_sparsity_op)



    def collect_quant_activation(self,layer_name,layer_idx,activation,FP_activation,
                                 weight,weight_spec,spec,digit_size,parallelism,
                                 in_features,out_features):
        if activation is None or spec is None or spec.kind == "none": return
        if self.ebb_stats is not None:
            context = dict(self.execution_context)
            context.update(self.attention_context.get((layer_name, layer_idx), {}))
            return self.ebb_stats.collect(layer_name, layer_idx, activation, weight,
                                          spec, weight_spec, in_features, out_features,
                                          self.current_phase, context)
        self.record_collected_layer_name(layer_name,layer_idx)
        sparse=sparse_counts(activation,spec,self.bit_scope,chunk_size=self.sparse_stat_chunk_size)
        self.collect_activation_sparsity(*sparse)
        self.collect_unit_sparsity(layer_name,layer_idx,activation,spec)
        attention=layer_name in {"qk_matmul","pv_matmul","qk_matmul_A","pv_matmul_A"}
        wt=sparse_counts(weight,weight_spec,"storage",chunk_size=self.sparse_stat_chunk_size)
        key=(self.current_phase,layer_name,layer_idx,weight_spec.name())
        if attention: self.collect_dynamic_weight_sparsity(*wt)
        elif key not in self._static_weights_seen:
            self.collect_weight_sparsity(*wt);self._static_weights_seen.add(key)
        cim=CIM_sys(h=self.cim_geometry["height"],w=self.cim_geometry["width"],
                    Nadder=self.cim_geometry["banks"],Nmacro=self.nmacro,freq=int(self.clock_freq_hz))
        mapped_activation=activation if attention else activation.reshape(-1,in_features)
        result=measure_mapping(cim,mapped_activation,spec,in_features,out_features,
                               self.current_phase!="decode",self.bit_scope,bool(self.as_l))
        result.update(layer_name=layer_name,layer_idx=layer_idx,phase=self.current_phase,
                      input_shape=list(activation.shape),
                      weight_format=weight_spec.name(),activation_format=spec.name(),
                      geometry=self.cim_geometry.copy(),weight_sparsity=dict(
                          elements=wt[0],zero_elements=wt[1],bits=wt[2],zero_bits=wt[3]),
                      activation_sparsity=dict(elements=sparse[0],zero_elements=sparse[1],
                                               bits=sparse[2],zero_bits=sparse[3]))
        result["sparse_compute_seconds"]=self._cycles_to_seconds(result["sparse_steps"])
        result["dense_compute_seconds"]=self._cycles_to_seconds(result["dense_steps"])
        self.cim_records.append(result)
        self._accumulate_cim_record(result)

    def _accumulate_cim_record(self,result):
        sparse=result["sparse_compute_seconds"];dense=result["dense_compute_seconds"]
        self.SACIM_latency_stat+=sparse;self.baseline_latency_stat+=dense
        self.latency+=sparse
        names={"q_proj":"q_proj","k_proj":"q_proj","v_proj":"q_proj",
               "qk_matmul":"qk_matmul","pv_matmul":"pv_matmul",
               "o_proj":"o_proj","out_proj":"o_proj","gate_proj":"gate_proj",
               "up_proj":"up_proj","down_proj":"down_proj"}
        prefix=names.get(result["layer_name"])
        if prefix:
            for suffix,value in [("SACIM_latency_stat",sparse),("baseline_latency_stat",dense)]:
                attr=prefix+"_"+suffix;setattr(self,attr,getattr(self,attr)+value)
        key=f"{result['phase']}:{result['layer_name']}_{result['layer_idx']}"
        entry=self.per_layer_latency.setdefault(key,dict(layer_name=result['layer_name'],
                  layer_idx=result['layer_idx'],phase=result['phase'],SACIM_latency=0.,baseline_latency=0.))
        entry['SACIM_latency']+=sparse;entry['baseline_latency']+=dense
        entry['speed_up']=entry['baseline_latency']/entry['SACIM_latency'] if entry['SACIM_latency']>0 else None

    def export_cim_stats(self,path):
        if self.ebb_stats is not None:
            raise ValueError("EBB results require export_ebb_stats, not the Asyn-CIM schema")
        import json
        from pathlib import Path
        doc=dict(schema_version=1,bit_scope=self.bit_scope,geometry=self.cim_geometry,
                 cycles_per_effective_bit=self.cycles_per_effective_bit,clock_freq_hz=self.clock_freq_hz,
                 scope="activation-bit compute only; no hidden-one, exponent, sign beyond selected scope, IO or weight-update costs",
                 records=self.cim_records)
        destination=Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(doc,indent=2,allow_nan=False)+"\n")


    def export_llmcompass_manifest(self,path,workload,source_commit):
        """Export the current LLMCompass capacity-baseline bridge ratios.

        This is one average per operator/phase, not a per-layer/context trace.
        Missing operators, all-zero costs and unmatched GQA are rejected.
        """
        if self.ebb_stats is not None:
            raise ValueError("The Asyn-CIM LLMCompass bridge cannot import EBB bounds")
        import json
        from pathlib import Path
        if self.as_l!=1: raise ValueError("LLMCompass import requires asynchronous mapping")
        if not source_commit or not isinstance(workload,dict): raise ValueError("Provenance is required")
        required={'d_model','ffn_dim','q_heads','kv_heads','batch_size','shared_kv_gqa',
                  'prefill_lengths','decode_cache_lengths'}
        if not required.issubset(workload): raise ValueError("Incomplete workload provenance")
        d=workload['d_model'];q=workload['q_heads'];kv=workload['kv_heads'];ffn=workload['ffn_dim']
        if min(d,q,kv,ffn,workload['batch_size'])<=0 or d%q or q%kv:
            raise ValueError("Invalid model dimensions")
        expected_shapes={'q_proj':(d,d),'k_proj':(d,d//q*kv),'v_proj':(d,d//q*kv),
                         'o_proj':(d,d),'out_proj':(d,d),'gate_proj':(d,ffn),
                         'up_proj':(d,ffn),'down_proj':(ffn,d)}
        for r in self.cim_records:
            if r['layer_name'] in expected_shapes:
                if (r['in_features'],r['out_features'])!=expected_shapes[r['layer_name']]:
                    raise ValueError("Projection dimensions differ from workload")
        observed_prefill=sorted({r['operand_shape'][-2]//workload['batch_size']
                                 for r in self.cim_records if r['phase']=='prefill' and r['layer_name']=='q_proj'})
        observed_decode=sorted({r['out_features']-1 for r in self.cim_records
                                if r['phase']=='decode' and r['layer_name']=='qk_matmul'})
        if observed_prefill!=workload['prefill_lengths'] or observed_decode!=workload['decode_cache_lengths']:
            raise ValueError("Observed prefill/decode contexts differ from workload")
        names={"q_proj":"Q_proj","k_proj":"K_proj","v_proj":"V_proj",
               "qk_matmul":"Q_mul_K","pv_matmul":"A_mul_V","o_proj":"H_matmul0",
               "out_proj":"H_matmul0","gate_proj":"Gate_proj","up_proj":"Up_proj","down_proj":"Down_proj"}
        speedups={};bits={}
        for phase in ("prefill","decode"):
            records=[r for r in self.cim_records if r['phase']==phase]
            widths={r['dense_bits'] for r in records}
            if len(widths)!=1: raise ValueError(f"{phase}: requires one dense bit width")
            bits[phase]=widths.pop();totals={}
            for r in records:
                if r['layer_name'] not in names: continue
                name=names[r['layer_name']]
                if phase=='decode' and name in ('Q_mul_K','A_mul_V'):
                    group=workload['q_heads']//workload['kv_heads'] if workload.get('shared_kv_gqa') else 1
                    if r['operand_shape'][-2]!=group:
                        raise ValueError("Decode attention grouping does not match shared-KV workload")
                    operands=r['operand_shape'][0]*r['operand_shape'][1]
                    expected=workload['batch_size']*(workload['kv_heads'] if workload.get('shared_kv_gqa') else workload['q_heads'])
                    if operands!=expected: raise ValueError("Decode KV/Q operand count mismatch")
                dense,sparse=totals.setdefault(name,[0.,0.])
                totals[name]=[dense+r['llmcompass_dense_steps'],sparse+r['sparse_steps']]
            if set(totals)!=set(names.values()): raise ValueError(f"{phase}: incomplete Qwen GEMM coverage")
            if any(sparse<=0 for _,sparse in totals.values()):
                raise ValueError("All-zero counted compute cannot be represented by a finite speedup")
            speedups[phase]={name:dense/sparse for name,(dense,sparse) in totals.items()}
        doc=dict(source_commit=source_commit,workload=workload,geometry=self.cim_geometry,
                 baseline='effective',dense_bits=bits,cycles_per_effective_bit=self.cycles_per_effective_bit,
                 activation_storage_bits=8,speedups=speedups,bit_scope=self.bit_scope,
                 ratio_semantics="LLMCompass capacity denominator / measured mapped bit steps",
                 approximation="one weighted operator average per phase; no layer/context trace",
                 weight_format="INT4 must be configured separately for LLMCompass memory traffic")
        destination=Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(doc,indent=2,allow_nan=False)+"\n")

    def _split_unit_layer_key(self, layer_key: str):  #add
        """  #add
        Examples:
            q_proj_0          -> q_proj, 0
            qk_matmul_A_0    -> qk_matmul_A, 0
            pv_matmul_B_27   -> pv_matmul_B, 27
        """  #add
        parts = layer_key.rsplit("_", 1)  #add
        if len(parts) == 2 and parts[1].isdigit():  #add
            return parts[0], int(parts[1])  #add
        return layer_key, -1  #add

    def export_unit_sparsity_csv(  #add
        self,  #add
        csv_path: str,  #add
        config_name: str = "",  #add
        model_path: str = "",  #add
    ):  #add
        import csv  #add
        import os  #add

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)  #add

        with open(csv_path, "w", newline="") as f:  #add
            writer = csv.writer(f)  #add
            writer.writerow([  #add
                "config",  #add
                "model_path",  #add
                "phase",  #add
                "layer_key",  #add
                "layer_type",  #add
                "layer_idx",  #add
                "zero_units",  #add
                "total_units",  #add
                "unit_zero_ratio",  #add
            ])  #add

            for phase in ["prefill", "decode"]:  #add
                phase_stats = self.unit_sparsity.get(phase, {})  #add

                for layer_key in sorted(phase_stats.keys()):  #add
                    counter = phase_stats[layer_key]  #add
                    zero = int(counter.get("zero_units", 0))  #add
                    total = int(counter.get("total_units", 0))  #add
                    ratio = zero / total if total > 0 else 0.0  #add

                    layer_type, layer_idx = self._split_unit_layer_key(layer_key)  #add

                    writer.writerow([  #add
                        config_name,  #add
                        model_path,  #add
                        phase,  #add
                        layer_key,  #add
                        layer_type,  #add
                        layer_idx,  #add
                        zero,  #add
                        total,  #add
                        ratio,  #add
                    ])  #add

    def export_unit_sparsity_summary_csv(  #add
        self,  #add
        csv_path: str,  #add
        config_name: str = "",  #add
        model_path: str = "",  #add
        append: bool = True,  #add
    ):  #add
        import csv  #add
        import os  #add

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)  #add

        file_exists = os.path.exists(csv_path)  #add
        mode = "a" if append else "w"  #add

        agg = {}  #add

        for phase in ["prefill", "decode"]:  #add
            phase_stats = self.unit_sparsity.get(phase, {})  #add

            for layer_key, counter in phase_stats.items():  #add
                layer_type, _ = self._split_unit_layer_key(layer_key)  #add
                key = (phase, layer_type)  #add

                if key not in agg:  #add
                    agg[key] = {  #add
                        "num_layers": 0,  #add
                        "zero_units": 0,  #add
                        "total_units": 0,  #add
                    }  #add

                agg[key]["num_layers"] += 1  #add
                agg[key]["zero_units"] += int(counter.get("zero_units", 0))  #add
                agg[key]["total_units"] += int(counter.get("total_units", 0))  #add

        with open(csv_path, mode, newline="") as f:  #add
            writer = csv.writer(f)  #add

            if (not file_exists) or (not append):  #add
                writer.writerow([  #add
                    "config",  #add
                    "model_path",  #add
                    "phase",  #add
                    "layer_type",  #add
                    "num_layers",  #add
                    "zero_units",  #add
                    "total_units",  #add
                    "unit_zero_ratio",  #add
                ])  #add

            for (phase, layer_type) in sorted(agg.keys()):  #add
                item = agg[(phase, layer_type)]  #add
                zero = item["zero_units"]  #add
                total = item["total_units"]  #add
                ratio = zero / total if total > 0 else 0.0  #add

                writer.writerow([  #add
                    config_name,  #add
                    model_path,  #add
                    phase,  #add
                    layer_type,  #add
                    item["num_layers"],  #add
                    zero,  #add
                    total,  #add
                    ratio,  #add
                ])  #add

    def print_collected_layer_names(  #add
        self,  #add
        phase: str = "full_forward",  #add
        title: str = None,  #add
    ):  #add
        if title is None:  #add
            title = f"{phase.upper()} COLLECTED LAYERS"  #add

        print("\n" + "-" * 80)  #add
        print(title)  #add
        print("-" * 80)  #add

        names = sorted(  #add
            self.collected_layer_names_by_phase.get(phase, set())  #add
        )  #add

        call_counts = self.collected_layer_call_count_by_phase.get(phase, {})  #add

        print(f"phase: {phase}")  #add
        print(f"unique collected layers: {len(names)}")  #add

        if len(names) == 0:  #add
            print("No collected layers.")  #add
            return  #add

        # Prefix summary, e.g. q_proj / qk_matmul_A / pv_matmul_B  #add
        prefix_counter = {}  #add
        for layer_key in names:  #add
            parts = layer_key.rsplit("_", 1)  #add
            if len(parts) == 2 and parts[1].isdigit():  #add
                prefix = parts[0]  #add
            else:  #add
                prefix = layer_key  #add
            prefix_counter[prefix] = prefix_counter.get(prefix, 0) + 1  #add

        print("\nLayer type summary:")  #add
        for prefix in sorted(prefix_counter.keys()):  #add
            print(f"  {prefix}: {prefix_counter[prefix]} layers")  #add

        print("\nCollected layer names:")  #add
        for layer_key in names:  #add
            print(  #add
                f"  {layer_key} "  #add
                f"(calls={call_counts.get(layer_key, 0)})"  #add
            )  #add

    def collect_unit_sparsity(self,layer_name,layer_idx,activation,spec):
        if not self.enable_unit_sparsity or activation is None or spec is None: return
        zero,total=unit_sparse_counts(activation,spec,self.bit_scope,
                                     self.unit_bit_group_size,self.unit_dim_group_size)
        counter=self._get_unit_counter(self.current_phase,f"{layer_name}_{layer_idx}")
        counter["zero_units"]+=zero;counter["total_units"]+=total


    def compute_unit_sparsity_int(self,tensor,bits,bit_group_size,dim_group_size,chunk_rows=2048):
        return unit_sparse_counts(tensor,parse_quant_spec(bits),"storage",bit_group_size,dim_group_size,chunk_rows)


    def _fp_stat_width(self, fmt: str):  #add
        """Return mantissa-only bit width for unit sparsity (0MMM convention).

        Matches the 0MMM codes from _fp_tensor_to_mantissa_fixed_chunk
        and _extract_sm_from_raw: only mantissa bits, no sign bit.
        """
        info = self._get_fp_format(fmt)  #add
        mant_bits = info["mant_bits"]  #add

        if fmt.lower().strip() in {"e4m3", "fp8_e4m3", "float8_e4m3"}:  #add
            return 3  # E4M3: mantissa only = 0MMM -> 3 bit  #add

        if fmt.lower().strip() in {"e2m1", "fp4", "fp4_e2m1"}:
            return 1  # E2M1: mantissa only = 0M -> 1 bit

        if fmt.lower().strip() in {"e5m10", "fp16", "float16"}:  #add
            return 10  # FP16/E5M10: mantissa only = 0MMMMMMMMMM -> 10 bit  #add

        if fmt.lower().strip() in {"bf16", "bfloat16"}:  #add
            return 7  # BF16: mantissa only = 0MMMMMMM -> 7 bit  #add

        raise ValueError(  #add
            f"Unsupported FP unit sparsity format {fmt}. "  #add
            f"Only E4M3, E5M10/FP16, and BF16 are supported."  #add
        )  #add

    def compute_unit_sparsity_fp(self,tensor,fmt,bit_group_size,dim_group_size,chunk_rows=2048):
        return unit_sparse_counts(tensor,parse_quant_spec(fmt),self.bit_scope,bit_group_size,dim_group_size,chunk_rows)


    def compute_sparse_stats(self,n,tensor,th=0.0,chunk_size=1_048_576):
        """Full n-bit two's-complement storage counts, including -8 in INT4."""
        return sparse_counts(tensor,parse_quant_spec(n),"storage",th,chunk_size)


    def _get_fp_format(self, fmt: str):
        """
        返回 FP 格式信息:
        total_bits: 原始 FP 总位宽
        exp_bits: exponent 位数
        mant_bits: fraction/mantissa 位数，不包不含 hidden leading 1
        """
        fmt = str(fmt).lower().strip()

        if fmt in {"e4m3", "fp8_e4m3", "float8_e4m3"}:
            return {
                "total_bits": 8,
                "exp_bits": 4,
                "mant_bits": 3,
            }

        if fmt in {"e2m1", "fp4", "fp4_e2m1"}:
            return {
                "total_bits": 4,
                "exp_bits": 2,
                "mant_bits": 1,
            }

        if fmt in {"e5m2", "fp8_e5m2", "float8_e5m2"}:
            return {
                "total_bits": 8,
                "exp_bits": 5,
                "mant_bits": 2,
            }

        if fmt in {"e5m10", "fp16", "float16"}:
            return {
                "total_bits": 16,
                "exp_bits": 5,
                "mant_bits": 10,
            }

        if fmt in {"bf16", "bfloat16"}:  #add
            return {  #add
                "total_bits": 16,  #add
                "exp_bits": 8,  #add
                "mant_bits": 7,  #add
            }  #add

        if fmt in {"fp32", "float32"}:  #add
            return {  #add
                "total_bits": 32,  #add
                "exp_bits": 8,  #add
                "mant_bits": 23,  #add
            }  #add

        raise ValueError(f"Unsupported FP format: {fmt}")

    def _fp_tensor_to_mantissa_fixed_chunk(
        self,
        tensor_chunk: torch.Tensor,
        fmt: str,
    ):
        """Return mantissa-only codes (0MMM format) used by the mapping model.

        The historical function name is kept for compatibility. No exponent
        shift and no hidden leading one are added; this matches ordinary
        ``Mapping_stat`` (0MMM for E4M3).
        """
        sm_codes, _, _ = self._encode_fp_values_to_sm_exp(tensor_chunk, fmt)
        return sm_codes

    def compute_sparse_stats_fp(self,fmt,tensor,th=0.0,chunk_size=1_048_576):
        return sparse_counts(tensor,parse_quant_spec(fmt),self.bit_scope,th,chunk_size)


    def save_all_scales(self):
        """Save all collected scales to files."""
        print(f"Saving scales to {self.scale_dir}...")

        for key, stat in self.stats.items():
            scales = stat.get_final_scales()

            # Save each scale type
            for scale_type, scale_value in scales.items():
                filename = f"{stat.layer_name}_{scale_type}_{stat.layer_idx}.p"
                filepath = os.path.join(self.scale_dir, filename)

                with open(filepath, 'wb') as f:
                    pickle.dump(scale_value, f)

                print(f"  Saved {filename}: {scale_value:.6f}")

        print(f"Total scales saved: {len(self.stats)} layers")

    def load_all_scales(self) -> Dict[str, Dict[str, float]]:
        """
        Load all scales from files.

        Returns:
            Dictionary mapping layer keys to their scales
        """
        loaded_scales = {}

        for key, stat in self.stats.items():
            scales = {}

            # Try to load each scale type
            for scale_type in ['w_scale', 'a_scale', 'o_scale', 'A_scale', 'B_scale', 'O_scale']:
                filename = f"{stat.layer_name}_{scale_type}_{stat.layer_idx}.p"
                filepath = os.path.join(self.scale_dir, filename)

                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        scales[scale_type] = pickle.load(f)

            if scales:
                loaded_scales[key] = scales

        return loaded_scales

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected statistics.

        Returns:
            Dictionary with summary information
        """
        summary = {
            'total_layers': len(self.stats),
            'total_samples': sum(stat.sample_count for stat in self.stats.values()),
            'layers': {}
        }

        for key, stat in self.stats.items():
            summary['layers'][key] = {
                'layer_name': stat.layer_name,
                'layer_idx': stat.layer_idx,
                'sample_count': stat.sample_count,
                'scales': stat.get_final_scales()
            }

        return summary

    def print_summary(self):
        """Print summary of collected statistics."""
        summary = self.get_summary()

        print("\n" + "="*80)
        print("Quantization Statistics Summary")
        print("="*80)
        print(f"Total layers: {summary['total_layers']}")
        print(f"Total samples: {summary['total_samples']}")
        print("\nPer-layer statistics:")
        print("-"*80)

        for key, info in summary['layers'].items():
            print(f"\n{key}:")
            print(f"  Samples: {info['sample_count']}")
            print(f"  Scales:")
            for scale_name, scale_value in info['scales'].items():
                print(f"    {scale_name}: {scale_value:.6f}")

        print("="*80 + "\n")

    def print_unit_sparsity_by_phase(self):  #add
        print("\n" + "-" * 80)  #add
        print("UNIT / BLOCK SPARSITY BY PHASE AND LAYER")  #add
        print("-" * 80)  #add

        print(  #add
            f"unit config: "  #add
            f"a(bit_group_size)={self.unit_bit_group_size}, "  #add
            f"b(dim_group_size)={self.unit_dim_group_size}"  #add
        )  #add

        for phase in ["prefill", "decode"]:  #add
            print(f"\n[{phase.upper()}]")  #add

            phase_stats = self.unit_sparsity.get(phase, {})  #add

            if not phase_stats:  #add
                print("  No unit sparsity statistics collected.")  #add
                continue  #add

            for layer_key in sorted(phase_stats.keys()):  #add
                counter = phase_stats[layer_key]  #add
                total = counter["total_units"]  #add
                zero = counter["zero_units"]  #add

                if total > 0:  #add
                    ratio = zero / total  #add
                    print(  #add
                        f"  {layer_key}: "  #add
                        f"zero_units={zero:,}, "  #add
                        f"total_units={total:,}, "  #add
                        f"unit_zero_ratio={ratio:.4%}"  #add
                    )  #add
                else:  #add
                    print(f"  {layer_key}: no valid units")  #add

    def reset_all(self):
        """Reset all statistics."""
        for stat in self.stats.values():
            stat.reset()

        # self.total_zero_count = 0  #delete
        # self.total_element_count = 0  #delete
        # self.total_bit_count = 0  #delete
        # self.total_0bit_count = 0  #delete
        # self.total_sparsebit_count = 0  #delete
        # self.total_amplitude_zero_bits_total = 0  #delete
        # self.total_amplitude_bit_count = 0  #delete

        self.reset_sparsity()  #add

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
