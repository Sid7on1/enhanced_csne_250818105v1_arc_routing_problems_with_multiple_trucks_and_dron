import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'agent': {
        'name': 'default_agent',
        'description': 'Default agent configuration'
    },
    'environment': {
        'name': 'default_environment',
        'description': 'Default environment configuration'
    },
    'metrics': {
        'enabled': True,
        'log_level': 'INFO'
    }
}

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class ConfigException(Exception):
    """Configuration exception"""
    pass

class Config(ABC):
    """Base configuration class"""
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration"""
        pass

    @abstractmethod
    def load(self, config_file: str) -> None:
        """Load configuration from file"""
        pass

    @abstractmethod
    def save(self, config_file: str) -> None:
        """Save configuration to file"""
        pass

class AgentConfig(Config):
    """Agent configuration class"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = config.get('name', DEFAULT_CONFIG['agent']['name'])
        self.description = config.get('description', DEFAULT_CONFIG['agent']['description'])

    def validate(self) -> None:
        """Validate agent configuration"""
        if not self.name:
            raise ConfigException('Agent name is required')

    def load(self, config_file: str) -> None:
        """Load agent configuration from file"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['agent']

    def save(self, config_file: str) -> None:
        """Save agent configuration to file"""
        with open(config_file, 'w') as f:
            yaml.dump({'agent': self.config}, f, default_flow_style=False)

class EnvironmentConfig(Config):
    """Environment configuration class"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.name = config.get('name', DEFAULT_CONFIG['environment']['name'])
        self.description = config.get('description', DEFAULT_CONFIG['environment']['description'])

    def validate(self) -> None:
        """Validate environment configuration"""
        if not self.name:
            raise ConfigException('Environment name is required')

    def load(self, config_file: str) -> None:
        """Load environment configuration from file"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['environment']

    def save(self, config_file: str) -> None:
        """Save environment configuration to file"""
        with open(config_file, 'w') as f:
            yaml.dump({'environment': self.config}, f, default_flow_style=False)

class MetricsConfig(Config):
    """Metrics configuration class"""
    def __init__(self, config: Dict):
        super().__init__(config)
        self.enabled = config.get('enabled', DEFAULT_CONFIG['metrics']['enabled'])
        self.log_level = LogLevel(config.get('log_level', DEFAULT_CONFIG['metrics']['log_level']))

    def validate(self) -> None:
        """Validate metrics configuration"""
        if not isinstance(self.enabled, bool):
            raise ConfigException('Metrics enabled must be a boolean')
        if not isinstance(self.log_level, LogLevel):
            raise ConfigException('Metrics log level must be a LogLevel enum')

    def load(self, config_file: str) -> None:
        """Load metrics configuration from file"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config['metrics']

    def save(self, config_file: str) -> None:
        """Save metrics configuration to file"""
        with open(config_file, 'w') as f:
            yaml.dump({'metrics': self.config}, f, default_flow_style=False)

class ConfigManager:
    """Configuration manager class"""
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.agent_config = AgentConfig({})
        self.environment_config = EnvironmentConfig({})
        self.metrics_config = MetricsConfig({})

    def load_config(self) -> None:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                self.agent_config.load(self.config_file)
                self.environment_config.load(self.config_file)
                self.metrics_config.load(self.config_file)
        else:
            logger.warning(f'Configuration file {self.config_file} not found')

    def save_config(self) -> None:
        """Save configuration to file"""
        self.agent_config.save(self.config_file)
        self.environment_config.save(self.config_file)
        self.metrics_config.save(self.config_file)

    def validate_config(self) -> None:
        """Validate configuration"""
        self.agent_config.validate()
        self.environment_config.validate()
        self.metrics_config.validate()

    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration"""
        return self.agent_config

    def get_environment_config(self) -> EnvironmentConfig:
        """Get environment configuration"""
        return self.environment_config

    def get_metrics_config(self) -> MetricsConfig:
        """Get metrics configuration"""
        return self.metrics_config

@contextmanager
def configure(config_file: str) -> ConfigManager:
    """Configure context manager"""
    config_manager = ConfigManager(config_file)
    try:
        config_manager.load_config()
        config_manager.validate_config()
        yield config_manager
    finally:
        config_manager.save_config()

if __name__ == '__main__':
    config_file = CONFIG_FILE
    with configure(config_file) as config_manager:
        agent_config = config_manager.get_agent_config()
        environment_config = config_manager.get_environment_config()
        metrics_config = config_manager.get_metrics_config()
        logger.info(f'Agent config: {agent_config.config}')
        logger.info(f'Environment config: {environment_config.config}')
        logger.info(f'Metrics config: {metrics_config.config}')