import {
  BaseNodeData,
  DropoutData,
  FeedForwardData,
  NormalizationData,
  PositionalEmbeddingData,
  TransformerBlockData,
  TestBlockData,
  TokenEmbeddingData,
  MHAttentionData,
  GQAttentionData,
} from './NodeData';
import { FieldConfig } from './FieldRenderer';
import { nodeFieldInfo } from './NodeInfo';
import ResidualLayer from '../Residual';
import MHAttentionLayer from '../MHAttention';
import GQAttentionLayer from '../GQAttention';
import TransformerBlock from '../TransformerBlock';
import FeedForwardLayer from '../FeedForward';
import DropoutLayer from '../Dropout';
import NormalizationLayer from '../Normalization';
import LinearOutputLayer from '../LinearOutput';
import TokenEmbeddingLayer from '../TokenEmbedding';
import PositionalEmbeddingLayer from '../PositionalEmbedding';
import TestBlock from '../TestBlock';
import { defaultConfig } from '../../Config';

interface NodeDefinition {
  // 1. 기본 정보
  type: string;
  label: string;

  // 2. 데이터 관련
  defaultData: BaseNodeData;

  configMapping?: {
    [key: string]: string;
  };

  // 3. 필드 관련
  stringFields: (keyof BaseNodeData)[];
  typeOptions?: Map<string, string[]>;
  getFields: (data: BaseNodeData) => FieldConfig[];

  // 4. 컴포넌트 관련
  component: React.ComponentType<any>;
}

export const getNodeTypes = () =>
  Object.fromEntries(
    Array.from(nodeRegistry.entries()).map(([type, def]) => [
      type,
      def.component,
    ]),
  );

export const getAllowedParentBlocks = () =>
  Array.from(nodeRegistry.values())
    .filter((def) => def.type.includes('Block'))
    .map((def) => def.type);

// Config로부터 Data를 받아 nodeType에 따라 node에 데이터 적용하는 함수
// 모든 노드의 모든 속성에 대해서 초기화 필요
export const getNodeDataByType = (
  nodeType: string,
  config: typeof defaultConfig,
  baseData: BaseNodeData,
): BaseNodeData => {
  const modelType = config.model;
  const data = { ...baseData, inDim: config.emb_dim, outDim: config.emb_dim };

  switch (modelType) {
    case 'gpt-2': {
      switch (nodeType) {
        case 'tokenEmbedding':
          return {
            ...data,
            vocabSize: config.vocab_size,
            embDim: config.emb_dim,
          };
        case 'positionalEmbedding':
          return {
            ...data,
            ctxLength: config.context_length,
            posType: 'Learned Positional Embedding', // Learned Positional Embedding, Sinusoidal Positional Embedding, Relative Positional Embedding, Rotary Positional Embedding
            vocabSize: config.vocab_size,
            embDim: config.emb_dim,
          };
        case 'feedForward':
          return {
            ...data,
            hiddenDim: 3072,
            feedForwardType: 'Standard', // Standard, Gated
            actFunc: 'GELU', // ReLU, GELU, SwiGLU, Mish
          };
        case 'linear':
          return {
            ...data,
            outDim: config.vocab_size, // 일단 Linear Output 기준으로 초기화
          };
        case 'dropout':
          return {
            ...data,
            dropoutRate: config.drop_rate,
          };
        case 'normalization':
          return {
            ...data,
            normType: 'Layer Normalization', // Layer Normalization, RMS Normalization
          };
        case 'mhAttention':
          return {
            ...data,
            dropoutRate: config.drop_rate,
            ctxLength: config.context_length,
            numHeads: config.n_heads,
            qkvBias: config.qkv_bias, // true, false
          };
        case 'gqAttention':
          return {
            ...data,
            dropoutRate: config.drop_rate,
            ctxLength: config.context_length,
            numHeads: config.n_heads,
            qkvBias: config.qkv_bias, // true, false
          };
        case 'transformerBlock':
          return {
            ...data,
            numOfBlocks: config.n_blocks,
          };
        default: // e.g., testBlock, residual
          return data;
      }
    }
    case 'llama2': {
      return data;
    }
    case 'llama3': {
      return data;
    }
    default: {
      return data;
    }
  }
};

export const nodeRegistry: Map<string, NodeDefinition> = new Map([
  [
    'testBlock',
    {
      type: 'testBlock',
      label: 'Test Block',
      component: TestBlock,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'Test Block',
      },
      stringFields: ['label'],
      getFields: (data: BaseNodeData) => {
        const typed = data as TestBlockData;
        return [
          {
            type: 'number',
            label: 'Test Type:',
            name: 'testType',
            value: typed.testType?.toString() || '',
            options: ['default', 'custom'],
            info: nodeFieldInfo.testBlock.testType,
          },
        ];
      },
    },
  ],
  [
    'tokenEmbedding',
    {
      type: 'tokenEmbedding',
      label: 'Token Embedding',
      component: TokenEmbeddingLayer,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'Token Embedding',
      },
      stringFields: ['label'],
      getFields: (data: BaseNodeData) => {
        const typed = data as TokenEmbeddingData;
        return [
          {
            type: 'number',
            label: 'Vocabulary Size:',
            name: 'vocabSize',
            value: typed.vocabSize?.toString() || '',
            placeholder: 'Enter vocabulary size',
            info: nodeFieldInfo.tokenEmbedding.vocabSize,
          },
          {
            type: 'number',
            label: 'Embedding Dimension Size:',
            name: 'embDim',
            value: typed.embDim?.toString() || '',
            placeholder: 'Enter embedding dimension',
            info: nodeFieldInfo.tokenEmbedding.embDim,
          },
        ];
      },
    },
  ],
  [
    'positionalEmbedding',
    {
      type: 'positionalEmbedding',
      label: 'Positional Embedding',
      component: PositionalEmbeddingLayer,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'Positional Embedding',
      },
      stringFields: ['label', 'posType'],
      typeOptions: new Map([
        [
          'posTypeOptions',
          [
            'Learned Positional Embedding',
            'Sinusoidal Positional Embedding',
            'Relative Positional Embedding',
            'Rotary Positional Embedding',
          ],
        ],
      ]),
      getFields: (data: BaseNodeData) => {
        const typed = data as PositionalEmbeddingData;
        return [
          {
            type: 'number',
            label: 'Context Length:',
            name: 'ctxLength',
            value: typed.ctxLength?.toString() || '',
            placeholder: 'Enter context length',
            info: nodeFieldInfo.positionalEmbedding.ctxLength,
          },
          {
            type: 'number',
            label: 'Embedding Dimension Size:',
            name: 'embDim',
            value: typed.embDim?.toString() || '',
            placeholder: 'Enter embedding dimension',
            info: nodeFieldInfo.positionalEmbedding.embDim,
          },
          {
            type: 'select',
            label: 'Positional Embedding Type:',
            name: 'posType',
            value: typed.posType || 'Learned Positional Embedding',
            options:
              nodeRegistry
                .get('positionalEmbedding')
                ?.typeOptions?.get('posTypeOptions') ?? [],
            info: nodeFieldInfo.positionalEmbedding.posType,
          },
        ];
      },
    },
  ],
  [
    'linear',
    {
      type: 'linear',
      label: 'Linear',
      component: LinearOutputLayer,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'Linear',
      },
      stringFields: ['label'],
      getFields: (data: BaseNodeData) => {
        return [
          {
            type: 'number',
            label: 'Output Dimension:',
            name: 'outDim',
            value: data.outDim?.toString() || '',
            placeholder: 'Enter output dimension',
            info: nodeFieldInfo.linear.outDim,
          },
        ];
      },
    },
  ],
  [
    'normalization',
    {
      type: 'normalization',
      label: 'Normalization',
      component: NormalizationLayer,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'Normalization',
      },
      stringFields: ['label', 'normType'],
      typeOptions: new Map([
        ['normTypeOptions', ['Layer Normalization', 'RMS Normalization']],
      ]),
      getFields: (data: BaseNodeData) => {
        const typed = data as NormalizationData;
        return [
          {
            type: 'select',
            label: 'Normalization Type:',
            name: 'normType',
            value: typed.normType || 'Layer Normalization',
            options:
              nodeRegistry
                .get('normalization')
                ?.typeOptions?.get('normTypeOptions') ?? [],
            info: nodeFieldInfo.normalization.normType,
          },
        ];
      },
    },
  ],
  [
    'feedForward',
    {
      type: 'feedForward',
      label: 'Feed Forward',
      component: FeedForwardLayer,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'Feed Forward',
      },
      stringFields: ['label', 'feedForwardType', 'actFunc'],
      typeOptions: new Map([
        ['feedForwardTypeOptions', ['Standard', 'Gated']],
        ['actFuncOptions', ['ReLU', 'GELU', 'SwiGLU', 'Mish']],
      ]),
      getFields: (data: BaseNodeData) => {
        const typed = data as FeedForwardData;
        return [
          {
            type: 'number',
            label: 'Hidden Dimension Size:',
            name: 'hiddenDim',
            value: typed.hiddenDim?.toString() || '',
            placeholder: 'Enter hidden dimension size',
            info: nodeFieldInfo.feedForward.hiddenDim,
          },
          {
            type: 'select',
            label: 'Feed Forward Type:',
            name: 'feedForwardType',
            value: typed.feedForwardType || 'Standard',
            options:
              nodeRegistry
                .get('feedForward')
                ?.typeOptions?.get('feedForwardTypeOptions') ?? [],
            info: nodeFieldInfo.feedForward.feedForwardType,
          },
          {
            type: 'select',
            label: 'Activation Function:',
            name: 'actFunc',
            value: typed.actFunc || 'GELU',
            options:
              nodeRegistry
                .get('feedForward')
                ?.typeOptions?.get('actFuncOptions') ?? [],
            info: nodeFieldInfo.feedForward.actFunc,
          },
        ];
      },
    },
  ],
  [
    'dropout',
    {
      type: 'dropout',
      label: 'Dropout',
      component: DropoutLayer,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'Dropout',
      },
      stringFields: ['label'],
      getFields: (data: BaseNodeData) => {
        const typed = data as DropoutData;
        return [
          {
            type: 'number',
            label: 'Dropout Rate:',
            name: 'dropoutRate',
            value: typed.dropoutRate?.toString() || '',
            placeholder: 'Enter dropout rate',
            info: nodeFieldInfo.dropout.dropoutRate,
          },
        ];
      },
    },
  ],
  [
    'residual',
    {
      type: 'residual',
      label: 'Residual',
      component: ResidualLayer,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'Residual',
      },
      stringFields: [],
      getFields: () => [],
    },
  ],
  [
    'mhAttention',
    {
      type: 'mhAttention',
      label: 'MH Attention',
      component: MHAttentionLayer,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'MH Attention',
      },
      stringFields: ['label'],
      getFields: (data: BaseNodeData) => {
        const typed = data as MHAttentionData;
        return [
          {
            type: 'number',
            label: 'Number of Heads:',
            name: 'numHeads',
            value: typed.numHeads?.toString() || '',
            placeholder: 'Enter number of heads',
            info: nodeFieldInfo.mhAttention.numHeads,
          },
          {
            type: 'number',
            label: 'Dropout Rate:',
            name: 'dropoutRate',
            value: typed.dropoutRate?.toString() || '',
            placeholder: 'Enter dropout rate',
            info: nodeFieldInfo.mhAttention.dropoutRate,
          },
          {
            type: 'select',
            label: 'QKV Bias:',
            name: 'qkvBias',
            value: typed.qkvBias ? 'true' : 'false',
            options: ['true', 'false'],
            info: nodeFieldInfo.mhAttention.qkvBias,
          },
        ];
      },
    },
  ],
  [
    'gqAttention',
    {
      type: 'gqAttention',
      label: 'GQ Attention',
      component: GQAttentionLayer,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'GQ Attention',
      },
      stringFields: ['label'],
      getFields: (data: BaseNodeData) => {
        const typed = data as GQAttentionData;
        return [
          {
            type: 'number',
            label: 'Number of Heads:',
            name: 'numHeads',
            value: typed.numHeads?.toString() || '',
            placeholder: 'Enter number of heads',
            info: nodeFieldInfo.gqAttention.numHeads,
          },
          {
            type: 'number',
            label: 'Context Length:',
            name: 'ctxLength',
            value: typed.ctxLength?.toString() || '',
            placeholder: 'Enter context length',
            info: nodeFieldInfo.gqAttention.ctxLength,
          },
          {
            type: 'number',
            label: 'Dropout Rate:',
            name: 'dropoutRate',
            value: typed.dropoutRate?.toString() || '',
            placeholder: 'Enter dropout rate',
            info: nodeFieldInfo.gqAttention.dropoutRate,
          },
          {
            type: 'select',
            label: 'QKV Bias:',
            name: 'qkvBias',
            value: typed.qkvBias ? 'true' : 'false',
            options: ['true', 'false'],
            info: nodeFieldInfo.gqAttention.qkvBias,
          },
        ];
      },
    },
  ],
  [
    'transformerBlock',
    {
      type: 'transformerBlock',
      label: 'Transformer Block',
      component: TransformerBlock,
      defaultData: {
        inDim: 0,
        outDim: 0,
        label: 'Transformer Block',
      },
      stringFields: ['label'],
      getFields: (data: BaseNodeData) => {
        const typed = data as TransformerBlockData;
        return [
          {
            type: 'number',
            label: 'Number of Blocks:',
            name: 'numOfBlocks',
            value: typed.numOfBlocks?.toString() || '',
            placeholder: 'Enter number of blocks',
            info: nodeFieldInfo.transformerBlock.numOfBlocks,
          },
        ];
      },
    },
  ],
]);
