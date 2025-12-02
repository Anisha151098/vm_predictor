"""
Enhanced Data Loader with 10x More Low CPU Minority Class Data
Based on realistic Azure VM behavior patterns
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple


class EnhancedAzureVMDataLoader:
    """Enhanced data loader with realistic Low CPU scenarios."""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def generate_low_cpu_vm(self, vm_id, n_timestamps, start_time, scenario_type):
        """
        Generate Low CPU VM based on real Azure scenarios.

        Scenarios:
        1. idle_vm: VM provisioned but barely used (common in dev/test)
        2. batch_job: Periodic spikes with long idle periods
        3. cache_server: Low CPU, high memory
        4. dns_server: Very low CPU, burst network
        5. failed_app: Stuck application, constant low CPU
        6. scheduled_task: Only active at specific times
        7. backup_server: Occasional activity
        8. monitoring_agent: Consistent minimal usage
        """
        timestamps = pd.date_range(start=start_time, periods=n_timestamps, freq='5min')
        t = np.arange(n_timestamps)

        if scenario_type == 'idle_vm':
            # Dev/test VM that's mostly idle (5-15% CPU)
            cpu_util = np.clip(10 + 5 * np.random.randn(n_timestamps), 0, 20)
            memory_util = np.clip(20 + 10 * np.random.randn(n_timestamps), 5, 40)
            network_util = np.clip(5 + 3 * np.random.randn(n_timestamps), 0, 15)

        elif scenario_type == 'batch_job':
            # Batch processing: 2-hour spikes every 12 hours, otherwise idle
            cpu_util = np.ones(n_timestamps) * 10
            spike_hours = [0, 12]  # Midnight and noon
            for hour in spike_hours:
                spike_start = (hour * 12) % n_timestamps  # 12 intervals per hour
                spike_duration = 24  # 2 hours
                if spike_start + spike_duration < n_timestamps:
                    cpu_util[spike_start:spike_start+spike_duration] = np.clip(
                        60 + 20 * np.random.randn(spike_duration), 40, 85
                    )
            cpu_util += np.random.normal(0, 3, n_timestamps)
            cpu_util = np.clip(cpu_util, 0, 85)
            memory_util = np.clip(30 + 10 * np.random.randn(n_timestamps), 15, 50)
            network_util = np.clip(15 + 8 * np.random.randn(n_timestamps), 5, 30)

        elif scenario_type == 'cache_server':
            # Redis/Memcached: Low CPU, high memory, moderate network
            cpu_util = np.clip(15 + 5 * np.sin(2*np.pi*t/288) + 5*np.random.randn(n_timestamps), 5, 30)
            memory_util = np.clip(75 + 10 * np.random.randn(n_timestamps), 60, 95)
            network_util = np.clip(40 + 15 * np.sin(2*np.pi*t/288) + 10*np.random.randn(n_timestamps), 20, 70)

        elif scenario_type == 'dns_server':
            # DNS/NTP server: Very low CPU, burst network
            cpu_util = np.clip(8 + 3 * np.random.randn(n_timestamps), 2, 18)
            memory_util = np.clip(25 + 8 * np.random.randn(n_timestamps), 10, 40)
            # Network bursts every ~hour
            network_util = np.ones(n_timestamps) * 10
            for burst in range(0, n_timestamps, 12):  # Every hour
                if burst + 2 < n_timestamps:
                    network_util[burst:burst+2] = np.clip(70 + 20*np.random.randn(2), 50, 95)
            network_util += np.random.normal(0, 5, n_timestamps)
            network_util = np.clip(network_util, 0, 95)

        elif scenario_type == 'failed_app':
            # Application stuck in error state: constant low CPU
            cpu_util = np.clip(12 + 2 * np.random.randn(n_timestamps), 8, 18)
            memory_util = np.clip(40 + 5 * np.random.randn(n_timestamps), 30, 50)
            network_util = np.clip(8 + 3 * np.random.randn(n_timestamps), 2, 15)

        elif scenario_type == 'scheduled_task':
            # Only active 9AM-5PM, idle otherwise
            cpu_util = np.ones(n_timestamps) * 8
            active_start = 9 * 12  # 9 AM (12 intervals per hour)
            active_end = 17 * 12   # 5 PM
            for day in range(n_timestamps // (24*12)):
                day_offset = day * 24 * 12
                start_idx = day_offset + active_start
                end_idx = min(day_offset + active_end, n_timestamps)
                if start_idx < n_timestamps:
                    duration = end_idx - start_idx
                    cpu_util[start_idx:end_idx] = np.clip(
                        25 + 10*np.sin(2*np.pi*np.arange(duration)/duration) + 8*np.random.randn(duration),
                        15, 45
                    )
            cpu_util += np.random.normal(0, 2, n_timestamps)
            cpu_util = np.clip(cpu_util, 0, 50)
            memory_util = np.clip(30 + 10 * np.random.randn(n_timestamps), 15, 50)
            network_util = np.clip(12 + 6 * np.random.randn(n_timestamps), 3, 30)

        elif scenario_type == 'backup_server':
            # Backup runs at 2AM, otherwise idle
            cpu_util = np.ones(n_timestamps) * 10
            backup_hour = 2 * 12  # 2 AM
            for day in range(n_timestamps // (24*12)):
                backup_start = day * 24 * 12 + backup_hour
                backup_duration = 36  # 3 hours
                if backup_start + backup_duration < n_timestamps:
                    cpu_util[backup_start:backup_start+backup_duration] = np.clip(
                        35 + 15*np.random.randn(backup_duration), 20, 60
                    )
            cpu_util += np.random.normal(0, 3, n_timestamps)
            cpu_util = np.clip(cpu_util, 0, 65)
            memory_util = np.clip(35 + 12 * np.random.randn(n_timestamps), 20, 60)
            network_util = np.clip(20 + 10 * np.random.randn(n_timestamps), 5, 50)

        else:  # monitoring_agent
            # Monitoring/logging agent: consistent minimal usage
            cpu_util = np.clip(12 + 3 * np.sin(2*np.pi*t/144) + 2*np.random.randn(n_timestamps), 8, 20)
            memory_util = np.clip(28 + 5 * np.random.randn(n_timestamps), 18, 40)
            network_util = np.clip(10 + 4 * np.random.randn(n_timestamps), 5, 20)

        # Create dataframe
        data = []
        for i in range(n_timestamps):
            data.append({
                'vm_id': f'{vm_id}_{scenario_type}',
                'timestamp': timestamps[i],
                'cpu_utilization': cpu_util[i],
                'memory_utilization': memory_util[i],
                'network_utilization': network_util[i],
                'scenario_type': scenario_type
            })

        return pd.DataFrame(data)

    def generate_high_cpu_vm(self, vm_id, n_timestamps, start_time, scenario_type):
        """
        Generate High CPU VM based on real Azure scenarios.

        Scenarios:
        1. web_server: Variable load with daily patterns
        2. database: High consistent usage
        3. compute_intensive: Near-constant high CPU
        4. ml_training: Sustained high load
        """
        timestamps = pd.date_range(start=start_time, periods=n_timestamps, freq='5min')
        t = np.arange(n_timestamps)

        if scenario_type == 'web_server':
            # Web server: Daily pattern with peak hours
            base_load = 50 + 25 * np.sin(2*np.pi*t/288)  # Daily cycle
            peak_hours = 15 * np.sin(2*np.pi*t/144)  # Lunch and evening peaks
            cpu_util = np.clip(base_load + peak_hours + 10*np.random.randn(n_timestamps), 30, 95)
            memory_util = np.clip(60 + 15 * np.random.randn(n_timestamps), 40, 85)
            network_util = np.clip(cpu_util * 0.8 + 10*np.random.randn(n_timestamps), 25, 90)

        elif scenario_type == 'database':
            # Database: High consistent usage with query spikes
            cpu_util = np.clip(70 + 10 * np.sin(2*np.pi*t/288) + 8*np.random.randn(n_timestamps), 50, 95)
            memory_util = np.clip(80 + 8 * np.random.randn(n_timestamps), 65, 95)
            network_util = np.clip(50 + 20 * np.random.randn(n_timestamps), 25, 85)

        elif scenario_type == 'compute_intensive':
            # Scientific computing/rendering: Near-constant high CPU
            cpu_util = np.clip(85 + 5 * np.random.randn(n_timestamps), 70, 98)
            memory_util = np.clip(75 + 10 * np.random.randn(n_timestamps), 55, 95)
            network_util = np.clip(30 + 15 * np.random.randn(n_timestamps), 10, 60)

        else:  # ml_training
            # ML training: Sustained high load with epochs
            epoch_duration = n_timestamps // 5  # 5 epochs
            cpu_util = np.ones(n_timestamps) * 80
            for epoch in range(5):
                start = epoch * epoch_duration
                end = min((epoch + 1) * epoch_duration, n_timestamps)
                # Ramp up, sustain, ramp down pattern
                epoch_pattern = np.concatenate([
                    np.linspace(60, 90, (end-start)//3),
                    np.ones((end-start)//3) * 90,
                    np.linspace(90, 60, (end-start)//3)
                ])
                if len(epoch_pattern) < (end-start):
                    epoch_pattern = np.pad(epoch_pattern, (0, (end-start)-len(epoch_pattern)), constant_values=80)
                cpu_util[start:end] = epoch_pattern[:(end-start)]
            cpu_util += np.random.normal(0, 5, n_timestamps)
            cpu_util = np.clip(cpu_util, 55, 98)
            memory_util = np.clip(85 + 5 * np.random.randn(n_timestamps), 70, 98)
            network_util = np.clip(25 + 10 * np.random.randn(n_timestamps), 10, 50)

        # Create dataframe
        data = []
        for i in range(n_timestamps):
            data.append({
                'vm_id': f'{vm_id}_{scenario_type}',
                'timestamp': timestamps[i],
                'cpu_utilization': cpu_util[i],
                'memory_utilization': memory_util[i],
                'network_utilization': network_util[i],
                'scenario_type': scenario_type
            })

        return pd.DataFrame(data)

    def generate_enhanced_dataset(
        self,
        n_low_cpu_vms=80,  # 10x more than before (was 8 out of 10)
        n_high_cpu_vms=20,
        n_timestamps=500,
        seed=42
    ):
        """
        Generate enhanced dataset with 10x more Low CPU VMs.

        Args:
            n_low_cpu_vms: Number of Low CPU VMs (80 for 80% Low CPU data)
            n_high_cpu_vms: Number of High CPU VMs (20 for 20% High CPU data)
            n_timestamps: Timestamps per VM
            seed: Random seed

        Returns:
            DataFrame with enhanced VM data
        """
        np.random.seed(seed)

        print("\n" + "="*70)
        print("GENERATING ENHANCED DATASET WITH 10X MORE LOW CPU DATA")
        print("="*70)

        all_data = []

        # Generate Low CPU VMs with diverse scenarios
        low_cpu_scenarios = [
            'idle_vm', 'batch_job', 'cache_server', 'dns_server',
            'failed_app', 'scheduled_task', 'backup_server', 'monitoring_agent'
        ]

        print(f"\nGenerating {n_low_cpu_vms} Low CPU VMs...")
        for i in range(n_low_cpu_vms):
            scenario = low_cpu_scenarios[i % len(low_cpu_scenarios)]
            start_time = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i//10)
            df_vm = self.generate_low_cpu_vm(f'low_cpu_{i}', n_timestamps, start_time, scenario)
            all_data.append(df_vm)
            if (i+1) % 20 == 0:
                print(f"  Generated {i+1}/{n_low_cpu_vms} Low CPU VMs...")

        # Generate High CPU VMs with diverse scenarios
        high_cpu_scenarios = ['web_server', 'database', 'compute_intensive', 'ml_training']

        print(f"\nGenerating {n_high_cpu_vms} High CPU VMs...")
        for i in range(n_high_cpu_vms):
            scenario = high_cpu_scenarios[i % len(high_cpu_scenarios)]
            start_time = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i//5)
            df_vm = self.generate_high_cpu_vm(f'high_cpu_{i}', n_timestamps, start_time, scenario)
            all_data.append(df_vm)

        # Combine all data
        df = pd.concat(all_data, ignore_index=True)

        # Add avg_cpu column for compatibility (threshold-based classification)
        df['avg_cpu'] = df['cpu_utilization'] / 100.0

        # Save to CSV
        filepath = os.path.join(self.data_dir, 'vm_usage_enhanced.csv')
        df.to_csv(filepath, index=False)

        # Print statistics
        low_cpu_count = (df['avg_cpu'] <= 0.5).sum()
        high_cpu_count = (df['avg_cpu'] > 0.5).sum()
        total = len(df)

        print(f"\n[OK] Enhanced dataset generated: {filepath}")
        print(f"  Total records: {total}")
        print(f"  VMs: {n_low_cpu_vms + n_high_cpu_vms}")
        print(f"  Timestamps per VM: {n_timestamps}")
        print(f"\n  Class Distribution (threshold=0.5):")
        print(f"    Low CPU (<=50%): {low_cpu_count} ({low_cpu_count/total*100:.1f}%)")
        print(f"    High CPU (>50%): {high_cpu_count} ({high_cpu_count/total*100:.1f}%)")
        print(f"\n  Low CPU Scenarios:")
        for scenario in low_cpu_scenarios:
            count = (df['scenario_type'] == scenario).sum()
            print(f"    - {scenario}: {count} records")
        print(f"\n  High CPU Scenarios:")
        for scenario in high_cpu_scenarios:
            count = (df['scenario_type'] == scenario).sum()
            print(f"    - {scenario}: {count} records")

        print("="*70)

        return df


if __name__ == "__main__":
    # Test enhanced data loader
    loader = EnhancedAzureVMDataLoader()

    # Generate enhanced dataset with 10x more Low CPU data
    df = loader.generate_enhanced_dataset(
        n_low_cpu_vms=80,    # 80 Low CPU VMs
        n_high_cpu_vms=20,   # 20 High CPU VMs
        n_timestamps=500,    # 500 timestamps per VM
        seed=42
    )

    print("\n[OK] Enhanced data loader test completed successfully!")
