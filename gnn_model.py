import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import List, Tuple, Dict

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
    
    def _load_cost_patterns(self) -> Dict:
        """Load known cost optimization patterns"""
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
        """Load known security patterns"""
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
        """Create PyTorch Geometric Data object"""
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        return Data(x=x, edge_index=edge_index_tensor)
    
    def predict_optimizations(self, graph_data: Data) -> Dict:
        """Predict cost and security optimizations"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(graph_data.x, graph_data.edge_index)
        
        # Extract predictions (cost_risk, security_risk, failure_risk)
        cost_risks = predictions[:, 0].numpy()
        security_risks = predictions[:, 1].numpy()
        failure_risks = predictions[:, 2].numpy()
        
        optimizations = {
            'cost_optimizations': self._analyze_cost_risks(cost_risks),
            'security_improvements': self._analyze_security_risks(security_risks),
            'reliability_fixes': self._analyze_failure_risks(failure_risks)
        }
        
        return optimizations
    
    def _analyze_cost_risks(self, cost_risks: np.ndarray) -> List[Dict]:
        """Analyze cost optimization opportunities"""
        optimizations = []
        high_cost_nodes = np.where(cost_risks > 0.7)[0]
        
        for node_idx in high_cost_nodes:
            optimizations.append({
                'node_index': int(node_idx),
                'type': 'cost_optimization',
                'severity': 'high' if cost_risks[node_idx] > 0.8 else 'medium',
                'recommendation': 'Consider rightsizing or alternative resource types',
                'potential_savings': f"{int(cost_risks[node_idx] * 30)}%"
            })
        
        return optimizations
    
    def _analyze_security_risks(self, security_risks: np.ndarray) -> List[Dict]:
        """Analyze security improvement opportunities"""
        improvements = []
        high_risk_nodes = np.where(security_risks > 0.6)[0]
        
        for node_idx in high_risk_nodes:
            improvements.append({
                'node_index': int(node_idx),
                'type': 'security_improvement',
                'severity': 'critical' if security_risks[node_idx] > 0.8 else 'high',
                'recommendation': 'Review security group rules and access policies',
                'risk_level': f"{int(security_risks[node_idx] * 100)}%"
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
        """Train model on synthetic data (placeholder for real training data)"""
        self.model.train()
        
        for epoch in range(num_epochs):
            # Generate synthetic training data
            synthetic_data = self._generate_synthetic_data()
            
            self.optimizer.zero_grad()
            predictions = self.model(synthetic_data.x, synthetic_data.edge_index)
            
            # Synthetic labels (in practice, use real optimization outcomes)
            labels = torch.randn(predictions.size())
            loss = F.mse_loss(predictions, labels)
            
            loss.backward()
            self.optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def _generate_synthetic_data(self) -> Data:
        """Generate synthetic graph data for training"""
        num_nodes = np.random.randint(5, 20)
        node_features = torch.randn(num_nodes, 5)
        
        # Create random edges
        num_edges = np.random.randint(num_nodes, num_nodes * 2)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        return Data(x=node_features, edge_index=edge_index)