import json
import yaml
import hcl2
from typing import Dict, List, Tuple
import networkx as nx

class IaCParser:
    def __init__(self):
        self.resource_types = {
            'aws_instance': {'category': 'compute', 'cost_weight': 0.8},
            'aws_rds_instance': {'category': 'database', 'cost_weight': 0.9},
            'aws_s3_bucket': {'category': 'storage', 'cost_weight': 0.3},
            'aws_security_group': {'category': 'security', 'cost_weight': 0.1},
            'aws_vpc': {'category': 'network', 'cost_weight': 0.2},
            'aws_subnet': {'category': 'network', 'cost_weight': 0.1},
        }
    
    def parse_terraform(self, tf_content: str) -> Dict:
        try:
            parsed = hcl2.loads(tf_content)
            resources = parsed.get('resource', [])
            
            all_resources = {}
            if isinstance(resources, list):
                for resource_block in resources:
                    if isinstance(resource_block, dict):
                        for resource_type, instances in resource_block.items():
                            if resource_type not in all_resources:
                                all_resources[resource_type] = {}
                            all_resources[resource_type].update(instances)
            elif isinstance(resources, dict):
                all_resources = resources
            
            return self._extract_resources(all_resources)
        except Exception as e:
            print(f"Error parsing Terraform: {e}")
            return {}
    
    def parse_cloudformation(self, cf_content: str) -> Dict:
        try:
            if cf_content.strip().startswith('{'):
                parsed = json.loads(cf_content)
            else:
                parsed = yaml.safe_load(cf_content)
            return self._extract_cf_resources(parsed.get('Resources', {}))
        except Exception as e:
            print(f"Error parsing CloudFormation: {e}")
            return {}
    
    def _extract_resources(self, resources: Dict) -> Dict:
        extracted = {}
        for resource_type, instances in resources.items():
            for name, config in instances.items():
                resource_id = f"{resource_type}.{name}"
                extracted[resource_id] = {
                    'type': resource_type,
                    'name': name,
                    'config': config,
                    'dependencies': self._find_dependencies(config),
                    'metadata': self.resource_types.get(resource_type, {'category': 'other', 'cost_weight': 0.5})
                }
        return extracted
    
    def _extract_cf_resources(self, resources: Dict) -> Dict:
        extracted = {}
        for name, resource in resources.items():
            resource_type = resource.get('Type', '').lower().replace('::', '_')
            extracted[name] = {
                'type': resource_type,
                'name': name,
                'config': resource.get('Properties', {}),
                'dependencies': self._find_cf_dependencies(resource),
                'metadata': self.resource_types.get(resource_type, {'category': 'other', 'cost_weight': 0.5})
            }
        return extracted
    
    def _find_dependencies(self, config: Dict) -> List[str]:
        deps = []
        config_str = str(config)
        import re
        refs = re.findall(r'\$\{([^}]+)\}', config_str)
        for ref in refs:
            if '.' in ref:
                deps.append(ref.split('.')[0] + '.' + ref.split('.')[1])
        return deps
    
    def _find_cf_dependencies(self, resource: Dict) -> List[str]:
        deps = []
        depends_on = resource.get('DependsOn', [])
        if isinstance(depends_on, str):
            deps.append(depends_on)
        elif isinstance(depends_on, list):
            deps.extend(depends_on)
        
        props_str = str(resource.get('Properties', {}))
        import re
        refs = re.findall(r'"Ref":\s*"([^"]+)"', props_str)
        deps.extend(refs)
        return deps

class GraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def build_graph(self, resources: Dict) -> nx.DiGraph:
        self.graph.clear()
        
        for resource_id, resource_data in resources.items():
            self.graph.add_node(resource_id, **resource_data)
        
        for resource_id, resource_data in resources.items():
            for dep in resource_data.get('dependencies', []):
                if dep in resources:
                    self.graph.add_edge(dep, resource_id)
        
        return self.graph
    
    def get_node_features(self) -> List[List[float]]:
        features = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            feature = [
                node_data['metadata']['cost_weight'],
                len(list(self.graph.predecessors(node))),
                len(list(self.graph.successors(node))),
                1.0 if node_data['metadata']['category'] == 'compute' else 0.0,
                1.0 if node_data['metadata']['category'] == 'security' else 0.0,
            ]
            features.append(feature)
        return features
    
    def get_edge_index(self) -> Tuple[List[int], List[int]]:
        node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}
        edge_index = [[], []]
        for edge in self.graph.edges():
            edge_index[0].append(node_to_idx[edge[0]])
            edge_index[1].append(node_to_idx[edge[1]])
        return edge_index