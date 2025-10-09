import numpy as np
import random
from typing import List, Dict, Tuple
from torch_geometric.data import Data
import torch

class SyntheticDataGenerator:
    def __init__(self):
        self.instance_types = ['t3.small', 't3.medium', 't3.large', 't3.xlarge', 't3.2xlarge']
        self.rds_types = ['db.t3.small', 'db.t3.medium', 'db.t3.large', 'db.t3.xlarge']
        
        self.utilization_patterns = {
            'underutilized': {'cpu_range': (5, 25), 'memory_range': (10, 30), 'should_optimize': True},
            'normal': {'cpu_range': (40, 70), 'memory_range': (50, 80), 'should_optimize': False},
            'high': {'cpu_range': (75, 95), 'memory_range': (80, 95), 'should_optimize': False}
        }
    
    def generate_training_batch(self, batch_size: int = 32) -> List[Dict]:
        batch = []
        for _ in range(batch_size):
            scenario = self._generate_single_scenario()
            batch.append(scenario)
        return batch
    
    def _generate_single_scenario(self) -> Dict:
        num_resources = random.randint(3, 12)
        resources = []
        
        for i in range(num_resources):
            resource_type = random.choice(['ec2', 'rds', 's3', 'vpc'])
            
            if resource_type == 'ec2':
                instance_type = random.choice(self.instance_types)
                pattern = random.choice(list(self.utilization_patterns.keys()))
                utilization = self.utilization_patterns[pattern]
                
                cpu_avg = random.uniform(*utilization['cpu_range'])
                memory_avg = random.uniform(*utilization['memory_range'])
                
                should_optimize = utilization['should_optimize'] and self._can_downsize(instance_type)
                target_type = self._get_smaller_instance(instance_type) if should_optimize else instance_type
                
                resources.append({
                    'id': f'ec2_{i}',
                    'type': 'aws_instance',
                    'instance_type': instance_type,
                    'cpu_avg': cpu_avg,
                    'memory_avg': memory_avg,
                    'should_optimize': should_optimize,
                    'target_type': target_type,
                    'dependencies': random.randint(0, 3)
                })
            
            elif resource_type == 'rds':
                instance_class = random.choice(self.rds_types)
                pattern = random.choice(list(self.utilization_patterns.keys()))
                utilization = self.utilization_patterns[pattern]
                
                cpu_avg = random.uniform(*utilization['cpu_range'])
                should_optimize = utilization['should_optimize'] and self._can_downsize_rds(instance_class)
                target_class = self._get_smaller_rds(instance_class) if should_optimize else instance_class
                
                resources.append({
                    'id': f'rds_{i}',
                    'type': 'aws_rds_instance',
                    'instance_class': instance_class,
                    'cpu_avg': cpu_avg,
                    'should_optimize': should_optimize,
                    'target_type': target_class,
                    'dependencies': random.randint(0, 2)
                })
            
            else:
                resources.append({
                    'id': f'{resource_type}_{i}',
                    'type': f'aws_{resource_type}',
                    'should_optimize': False,
                    'dependencies': random.randint(0, 1)
                })
        
        return {'resources': resources}
    
    def _can_downsize(self, instance_type: str) -> bool:
        downsizable = ['t3.medium', 't3.large', 't3.xlarge', 't3.2xlarge']
        return instance_type in downsizable
    
    def _get_smaller_instance(self, instance_type: str) -> str:
        mapping = {
            't3.medium': 't3.small',
            't3.large': 't3.medium',
            't3.xlarge': 't3.large',
            't3.2xlarge': 't3.xlarge'
        }
        return mapping.get(instance_type, instance_type)
    
    def _can_downsize_rds(self, instance_class: str) -> bool:
        downsizable = ['db.t3.medium', 'db.t3.large', 'db.t3.xlarge']
        return instance_class in downsizable
    
    def _get_smaller_rds(self, instance_class: str) -> str:
        mapping = {
            'db.t3.medium': 'db.t3.small',
            'db.t3.large': 'db.t3.medium',
            'db.t3.xlarge': 'db.t3.large'
        }
        return mapping.get(instance_class, instance_class)
    
    def scenario_to_graph_data(self, scenario: Dict) -> Tuple[Data, List[int]]:
        resources = scenario['resources']
        node_features = []
        labels = []
        
        for resource in resources:
            cpu_util = resource.get('cpu_avg', 50) / 100.0
            memory_util = resource.get('memory_avg', 50) / 100.0
            deps = min(resource.get('dependencies', 0) / 5.0, 1.0)
            is_compute = 1.0 if resource['type'] in ['aws_instance', 'aws_rds_instance'] else 0.0
            
            cost_weights = {
                'aws_instance': 0.8,
                'aws_rds_instance': 0.9,
                'aws_s3': 0.3,
                'aws_vpc': 0.1
            }
            cost_weight = cost_weights.get(resource['type'], 0.5)
            
            features = [cost_weight, cpu_util, memory_util, deps, is_compute]
            node_features.append(features)
            
            labels.append(1 if resource.get('should_optimize', False) else 0)
        
        edge_index = [[], []]
        for i in range(len(resources) - 1):
            edge_index[0].append(i)
            edge_index[1].append(i + 1)
        
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index_tensor), labels