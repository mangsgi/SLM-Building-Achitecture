import { nodeInformation } from '../../constants/nodeInformation';
import { nodeAttributeInformation } from '../../constants/nodeAttributeInformation';

// 노드 정보 인터페이스
export interface NodeInfo {
  title: string;
  description: string;
}

// 필드 정보 인터페이스
export interface FieldInfo {
  title: string;
  description: string;
}

// 노드별 필드 정보 타입
export interface NodeFieldInfo {
  [key: string]: FieldInfo;
}

// 노드 정보 저장소
export const nodeInfo: { [key: string]: NodeInfo } = {
  positionalEmbedding: {
    title: 'Positional Embedding',
    description: nodeInformation.positionalEmbedding.description,
  },
  tokenEmbedding: {
    title: 'Token Embedding',
    description: nodeInformation.tokenEmbedding.description,
  },
  normalization: {
    title: 'Normalization',
    description: nodeInformation.normalization.description,
  },
  feedForward: {
    title: 'Feed Forward',
    description: nodeInformation.feedForward.description,
  },
  dropout: {
    title: 'Dropout',
    description: nodeInformation.dropout.description,
  },
  linear: {
    title: 'Linear Output',
    description: nodeInformation.linear.description,
  },
  mhAttention: {
    title: 'Multi-Head Attention',
    description: nodeInformation.mhAttention.description,
  },
  transformerBlock: {
    title: 'Transformer Block',
    description: nodeInformation.transformerBlock.description,
  },
  dynamicBlock: {
    title: 'Dynamic Block',
    description: nodeInformation.dynamicBlock.description,
  },
  residual: {
    title: 'Residual Connection',
    description: nodeInformation.residual.description,
  },
  testBlock: {
    title: 'Test Block',
    description: nodeInformation.testBlock.description,
  },
  gqAttention: {
    title: 'Grouped Query Attention',
    description: nodeInformation.gqAttention.description,
  },
};

// 노드별 필드 정보 저장소
export const nodeFieldInfo: { [key: string]: NodeFieldInfo } = {
  positionalEmbedding: {
    ctxLength: {
      title: 'Context Length',
      description:
        nodeAttributeInformation.positionalEmbedding.ctxLength.description,
    },
    embDim: {
      title: 'Embedding Dimension Size',
      description:
        nodeAttributeInformation.positionalEmbedding.embDim.description,
    },
    posType: {
      title: 'Positional Embedding Type',
      description:
        nodeAttributeInformation.positionalEmbedding.posType.description,
    },
  },
  tokenEmbedding: {
    vocabSize: {
      title: 'Vocabulary Size',
      description:
        nodeAttributeInformation.tokenEmbedding.vocabSize.description,
    },
    embDim: {
      title: 'Embedding Dimension Size',
      description: nodeAttributeInformation.tokenEmbedding.embDim.description,
    },
  },
  normalization: {
    normType: {
      title: 'Normalization Type',
      description: nodeAttributeInformation.normalization.normType.description,
    },
    eps: {
      title: 'Epsilon',
      description: nodeAttributeInformation.normalization.eps.description,
    },
    inDim: {
      title: 'Input Dimension',
      description: nodeAttributeInformation.normalization.inDim.description,
    },
  },
  feedForward: {
    actFunc: {
      title: 'Activation Function',
      description: nodeAttributeInformation.feedForward.actFunc.description,
    },
    feedForwardType: {
      title: 'Feed Forward Type',
      description:
        nodeAttributeInformation.feedForward.feedForwardType.description,
    },
    hiddenDim: {
      title: 'Hidden Dimension Size',
      description: nodeAttributeInformation.feedForward.hiddenDim.description,
    },
    bias: {
      title: 'Bias Enabled',
      description: nodeAttributeInformation.feedForward.bias.description,
    },
  },
  dropout: {
    dropoutRate: {
      title: 'Dropout Rate',
      description: nodeAttributeInformation.dropout.dropoutRate.description,
    },
  },
  linear: {
    outDim: {
      title: 'Output Dimension',
      description: nodeAttributeInformation.linear.outDim.description,
    },
    weightTying: {
      title: 'Weight Tying',
      description: nodeAttributeInformation.linear.weightTying.description,
    },
    bias: {
      title: 'Bias Enabled',
      description: nodeAttributeInformation.linear.bias.description,
    },
  },
  mhAttention: {
    numHeads: {
      title: 'Number of Heads',
      description: nodeAttributeInformation.mhAttention.numHeads.description,
    },
    dropoutRate: {
      title: 'Attention Dropout Rate',
      description: nodeAttributeInformation.mhAttention.dropoutRate.description,
    },
    qkvBias: {
      title: 'QKV Bias',
      description: nodeAttributeInformation.mhAttention.qkvBias.description,
    },
    isRoPE: {
      title: 'RoPE Enabled',
      description: nodeAttributeInformation.mhAttention.isRoPE.description,
    },
    ropeBase: {
      title: 'Rope Base',
      description: nodeAttributeInformation.mhAttention.ropeBase.description,
    },
  },
  gqAttention: {
    numHeads: {
      title: 'Number of Heads',
      description: nodeAttributeInformation.gqAttention.numHeads.description,
    },
    ctxLength: {
      title: 'Context Length',
      description: nodeAttributeInformation.gqAttention.ctxLength.description,
    },
    dropoutRate: {
      title: 'Attention Dropout Rate',
      description: nodeAttributeInformation.gqAttention.dropoutRate.description,
    },
    qkvBias: {
      title: 'QKV Bias',
      description: nodeAttributeInformation.gqAttention.qkvBias.description,
    },
    isRoPE: {
      title: 'RoPE Enabled',
      description: nodeAttributeInformation.gqAttention.isRoPE.description,
    },
    ropeBase: {
      title: 'Rope Base',
      description: nodeAttributeInformation.gqAttention.ropeBase.description,
    },
    qkNorm: {
      title: 'QK Normalization',
      description: nodeAttributeInformation.gqAttention.qkNorm.description,
    },
    qkNormEps: {
      title: 'QK Normalization Eps',
      description: nodeAttributeInformation.gqAttention.qkNormEps.description,
    },
  },
  transformerBlock: {
    numOfBlocks: {
      title: 'Number of Blocks',
      description:
        nodeAttributeInformation.transformerBlock.numOfBlocks.description,
    },
  },
  dynamicBlock: {
    numOfBlocks: {
      title: 'Number of Blocks',
      description:
        nodeAttributeInformation.dynamicBlock.numOfBlocks.description,
    },
  },
  testBlock: {
    testType: {
      title: 'Test Type',
      description: nodeAttributeInformation.testBlock.testType.description,
    },
  },
};
