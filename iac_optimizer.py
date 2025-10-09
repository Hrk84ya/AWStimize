from iac_parser import IaCParser, GraphBuilder
from gnn_model import OptimizationPredictor
from aws_pricing import AWSPricing
import json
from typing import Dict, List

class IaCOptimizer:
    def __init__(self):
        self.parser = IaCParser()
        self.graph_builder = GraphBuilder()
        self.predictor = OptimizationPredictor()
        self.pricing = AWSPricing()
        
    def analyze_terraform(self, tf_content: str) -> Dict:
        """Analyze Terraform configuration and suggest optimizations"""
        resources = self.parser.parse_terraform(tf_content)
        return self._analyze_resources(resources)
    
    def analyze_cloudformation(self, cf_content: str) -> Dict:
        """Analyze CloudFormation template and suggest optimizations"""
        resources = self.parser.parse_cloudformation(cf_content)
        return self._analyze_resources(resources)
    
    def _analyze_resources(self, resources: Dict) -> Dict:
        """Core analysis logic"""
        if not resources:
            return {"error": "No resources found or parsing failed"}
        
        # Build dependency graph
        graph = self.graph_builder.build_graph(resources)
        
        # Extract features for GNN
        node_features = self.graph_builder.get_node_features()
        edge_index = self.graph_builder.get_edge_index()
        
        # Create graph data
        graph_data = self.predictor.create_graph_data(node_features, edge_index)
        
        # Calculate costs first
        cost_analysis = self._calculate_costs(resources)
        
        # Get predictions
        optimizations = self.predictor.predict_optimizations(graph_data)
        
        # Add real cost optimizations from pricing analysis
        real_cost_opts = []
        for item in cost_analysis['breakdown']:
            if item['rightsizing']:
                rs = item['rightsizing']
                real_cost_opts.append({
                    'resource_name': item['resource'],
                    'resource_type': item['type'],
                    'type': 'cost_optimization',
                    'severity': 'medium',
                    'recommendation': f"Rightsize from {rs['current_type']} to {rs['suggested_type']}",
                    'potential_savings': f"${rs['monthly_savings']}/month ({rs['savings_percentage']}%)"
                })
        
        # Merge real cost optimizations with GNN predictions
        optimizations['cost_optimizations'].extend(real_cost_opts)
        
        # Add resource context to GNN predictions
        resource_list = list(resources.keys())
        for opt_type in optimizations:
            for opt in optimizations[opt_type]:
                if 'node_index' in opt and opt['node_index'] < len(resource_list):
                    opt['resource_name'] = resource_list[opt['node_index']]
                    opt['resource_type'] = resources[resource_list[opt['node_index']]]['type']
        

        
        # Calculate optimized cost
        optimized_cost = cost_analysis['total_cost'] - cost_analysis['total_savings']
        
        # Add graph analysis
        analysis_result = {
            'summary': {
                'total_resources': len(resources),
                'total_dependencies': len(graph.edges()),
                'complexity_score': len(graph.edges()) / max(len(resources), 1),
                'estimated_monthly_cost': cost_analysis['total_cost'],
                'optimized_monthly_cost': round(optimized_cost, 2),
                'total_monthly_savings': cost_analysis['total_savings']
            },
            'cost_breakdown': cost_analysis['breakdown'],
            'optimizations': optimizations,
            'graph_metrics': self._calculate_graph_metrics(graph),
            'recommendations': self._generate_recommendations(optimizations)
        }
        
        return analysis_result
    
    def _calculate_costs(self, resources: Dict) -> Dict:
        """Calculate infrastructure costs with optimization suggestions"""
        total_cost = 0.0
        total_savings = 0.0
        breakdown = []
        
        for resource_id, resource_data in resources.items():
            resource_type = resource_data['type']
            config = resource_data['config']
            
            cost = self.pricing.calculate_resource_cost(resource_type, config)
            total_cost += cost
            
            # Get rightsizing recommendation
            rightsizing = self.pricing.get_rightsizing_recommendation(resource_type, config)
            if rightsizing:
                total_savings += rightsizing['monthly_savings']
            
            breakdown.append({
                'resource': resource_id,
                'type': resource_type,
                'monthly_cost': round(cost, 2),
                'rightsizing': rightsizing
            })
        
        return {
            'total_cost': round(total_cost, 2),
            'total_savings': round(total_savings, 2),
            'breakdown': breakdown
        }
    
    def _calculate_graph_metrics(self, graph) -> Dict:
        """Calculate graph complexity metrics"""
        import networkx as nx
        
        metrics = {
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_dag': nx.is_directed_acyclic_graph(graph)
        }
        
        if graph.number_of_nodes() > 0:
            try:
                metrics['avg_clustering'] = nx.average_clustering(graph.to_undirected())
            except:
                metrics['avg_clustering'] = 0.0
        
        return metrics
    
    def _generate_recommendations(self, optimizations: Dict) -> List[str]:
        """Generate high-level recommendations"""
        recommendations = []
        
        cost_opts = len(optimizations.get('cost_optimizations', []))
        security_opts = len(optimizations.get('security_improvements', []))
        reliability_opts = len(optimizations.get('reliability_fixes', []))
        
        if cost_opts > 0:
            recommendations.append(f"Found {cost_opts} cost optimization opportunities")
        
        if security_opts > 0:
            recommendations.append(f"Identified {security_opts} security improvements needed")
        
        if reliability_opts > 0:
            recommendations.append(f"Detected {reliability_opts} reliability risks to address")
        
        if not recommendations:
            recommendations.append("Infrastructure appears well-optimized")
        
        return recommendations
    
    def train_model(self):
        """Train the GNN model (placeholder for production training)"""
        print("Training GNN model on synthetic data...")
        self.predictor.train_on_synthetic_data(num_epochs=50)
        print("Training completed")

def main():
    """Example usage"""
    optimizer = IaCOptimizer()
    
    # Train model first
    optimizer.train_model()
    
    # Example Terraform configuration
    terraform_example = '''
    resource "aws_instance" "web" {
      ami           = "ami-0c02fb55956c7d316"
      instance_type = "t3.large"
      
      vpc_security_group_ids = [aws_security_group.web.id]
      subnet_id = aws_subnet.public.id
    }
    
    resource "aws_security_group" "web" {
      name = "web-sg"
      
      ingress {
        from_port   = 80
        to_port     = 80
        protocol    = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
      }
    }
    
    resource "aws_subnet" "public" {
      vpc_id = aws_vpc.main.id
      cidr_block = "10.0.1.0/24"
    }
    
    resource "aws_vpc" "main" {
      cidr_block = "10.0.0.0/16"
    }
    '''
    
    # Analyze configuration
    result = optimizer.analyze_terraform(terraform_example)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()