import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import List, Tuple, Dict
from synthetic_data_generator import SyntheticDataGenerator

class IaCGNN(torch.nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, output_dim: int = 3):
        super(IaCGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

class OptimizationPredictor:
    def __init__(self):
        self.model = IaCGNN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.cost_patterns = self._load_cost_patterns()
        self.security_patterns = self._load_security_patterns()
        self.data_generator = SyntheticDataGenerator()
        self.training_accuracy = 0.0
    
    def _load_cost_patterns(self) -> Dict:
        return {
            'oversized_instances': {
                'pattern': 'high_cost_low_utilization',
                'recommendation': 'Consider smaller instance types',
                'savings': 0.3
            },
            'unused_resources': {
                'pattern': 'zero_dependencies',
                'recommendation': 'Remove unused resources',
                'savings': 1.0
            },
            'redundant_storage': {
                'pattern': 'multiple_s3_buckets',
                'recommendation': 'Consolidate storage buckets',
                'savings': 0.2
            }
        }
    
    def _load_security_patterns(self) -> Dict:
        return {
            'open_security_groups': {
                'pattern': 'sg_0.0.0.0/0',
                'risk': 'high',
                'recommendation': 'Restrict security group rules'
            },
            'unencrypted_storage': {
                'pattern': 'no_encryption',
                'risk': 'medium',
                'recommendation': 'Enable encryption at rest'
            },
            'public_subnets': {
                'pattern': 'public_subnet_db',
                'risk': 'high',
                'recommendation': 'Move databases to private subnets'
            }
        }
    
    def create_graph_data(self, node_features: List[List[float]], edge_index: Tuple[List[int], List[int]]) -> Data:
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        return Data(x=x, edge_index=edge_index_tensor)
    
    def predict_optimizations(self, graph_data: Data, resources: Dict) -> Dict:
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(graph_data.x, graph_data.edge_index)
        
        cost_predictions = torch.sigmoid(predictions[:, 0]).numpy()
        
        optimizations = {
            'cost_optimizations': self._analyze_ml_predictions(cost_predictions, resources),
            'security_improvements': self._analyze_security_patterns(resources),
            'reliability_fixes': []
        }
        
        return optimizations
    
    def _analyze_ml_predictions(self, predictions: np.ndarray, resources: Dict) -> List[Dict]:
        optimizations = []
        resource_list = list(resources.keys())
        
        for idx, prediction in enumerate(predictions):
            if idx < len(resource_list) and prediction > 0.5:
                resource_id = resource_list[idx]
                resource_data = resources[resource_id]
                
                if resource_data['type'] in ['aws_instance', 'aws_rds_instance']:
                    optimizations.append({
                        'node_index': idx,
                        'resource_name': resource_id,
                        'resource_type': resource_data['type'],
                        'type': 'ml_cost_optimization',
                        'severity': 'high' if prediction > 0.8 else 'medium',
                        'recommendation': f'ML model suggests optimization (confidence: {prediction:.1%})',
                        'ml_confidence': f'{prediction:.1%}'
                    })
        
        return optimizations
    
    def _analyze_security_patterns(self, resources: Dict) -> List[Dict]:
        improvements = []
        
        for resource_id, resource_data in resources.items():
            config = resource_data.get('config', {})
            
            if resource_data['type'] == 'aws_security_group':
                ingress_rules = config.get('ingress', [])
                if not isinstance(ingress_rules, list):
                    ingress_rules = [ingress_rules]
                
                for rule in ingress_rules:
                    cidr_blocks = rule.get('cidr_blocks', [])
                    if '0.0.0.0/0' in cidr_blocks:
                        improvements.append({
                            'resource_name': resource_id,
                            'resource_type': resource_data['type'],
                            'type': 'security_improvement',
                            'severity': 'high',
                            'recommendation': 'Restrict overly permissive security group rules',
                            'issue': 'Open to all IPs (0.0.0.0/0)'
                        })
            
            elif resource_data['type'] == 'aws_rds_instance':
                if not config.get('storage_encrypted', True):
                    improvements.append({
                        'resource_name': resource_id,
                        'resource_type': resource_data['type'],
                        'type': 'security_improvement',
                        'severity': 'medium',
                        'recommendation': 'Enable storage encryption',
                        'issue': 'Storage not encrypted'
                    })
        
        return improvements
    
    def _analyze_failure_risks(self, failure_risks: np.ndarray) -> List[Dict]:
        """Analyze cascade failure risks"""
        fixes = []
        high_failure_nodes = np.where(failure_risks > 0.7)[0]
        
        for node_idx in high_failure_nodes:
            fixes.append({
                'node_index': int(node_idx),
                'type': 'reliability_fix',
                'severity': 'high',
                'recommendation': 'Add redundancy or implement circuit breakers',
                'failure_probability': f"{int(failure_risks[node_idx] * 100)}%"
            })
        
        return fixes
    
    def train_on_synthetic_data(self, num_epochs: int = 100):
        self.model.train()
        total_correct = 0
        total_samples = 0
        
        for epoch in range(num_epochs):
            batch_scenarios = self.data_generator.generate_training_batch(batch_size=16)
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for scenario in batch_scenarios:
                graph_data, labels = self.data_generator.scenario_to_graph_data(scenario)
                
                self.optimizer.zero_grad()
                predictions = self.model(graph_data.x, graph_data.edge_index)
                
                cost_predictions = predictions[:, 0]
                labels_tensor = torch.tensor(labels, dtype=torch.float)
                
                loss = F.binary_cross_entropy_with_logits(cost_predictions, labels_tensor)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                predicted_labels = (torch.sigmoid(cost_predictions) > 0.5).float()
                correct = (predicted_labels == labels_tensor).sum().item()
                epoch_correct += correct
                epoch_samples += len(labels)
            
            total_correct += epoch_correct
            total_samples += epoch_samples
            
            if epoch % 20 == 0:
                accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0
                print(f"Epoch {epoch}, Loss: {epoch_loss/len(batch_scenarios):.4f}, Accuracy: {accuracy:.3f}")
        
        self.training_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"\nFinal Training Accuracy: {self.training_accuracy:.3f}")
    
    def get_model_accuracy(self) -> float:
        return self.training_accuracy