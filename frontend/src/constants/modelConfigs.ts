// --- 타입 정의 시작 ---
import { RopeConfig } from '../nodes/components/NodeData';

export type ModelType = 'GPT-2' | 'Llama2' | 'Llama3' | 'Qwen3' | 'SmolLM3';

// 공통 설정값 인터페이스
interface BaseConfig {
  epochs: number;
  batch_size: number;
  vocab_size: number;
  context_length: number;
  stride: number;
  emb_dim: number;
  n_heads: number;
  n_blocks: number;
  dtype: string;
}

// 모델별 고유 설정값 인터페이스
export interface GPT2Config extends BaseConfig {
  model: 'gpt-2';
  drop_rate: number;
  qkv_bias: boolean;
}

export interface Llama2Config extends BaseConfig {
  model: 'llama2';
  hidden_dim: number;
}

export interface Llama3Config extends BaseConfig {
  model: 'llama3';
  hidden_dim: number;
  n_kv_groups: number;
  rope_base: number;
  rope_freq: RopeConfig;
}

export interface Qwen3Config extends BaseConfig {
  model: 'qwen3';
  hidden_dim: number;
  head_dim: number;
  qk_norm: boolean;
  n_kv_groups: number;
  rope_base: number;
}

export interface SmolLM3Config extends BaseConfig {
  model: 'smollm3';
  hidden_dim: number;
}

// 구별된 유니온 타입
export type ModelConfig =
  | GPT2Config
  | Llama2Config
  | Llama3Config
  | Qwen3Config
  | SmolLM3Config;
// --- 타입 정의 끝 ---

// --- 설정값 객체들 ---
const gpt2Config: Omit<GPT2Config, 'model'> = {
  // 128M GPT-2
  epochs: 1,
  batch_size: 1,
  vocab_size: 50257,
  context_length: 128, // 1024
  stride: 128, // == context_length
  emb_dim: 768,
  n_heads: 12,
  n_blocks: 12,
  drop_rate: 0.1,
  qkv_bias: true,
  dtype: 'bf16',
};

const llama2Config: Omit<Llama2Config, 'model'> = {
  // 7B Llama2
  epochs: 1,
  batch_size: 1,
  vocab_size: 32000,
  context_length: 128, // 4096
  stride: 128, // == context_length
  emb_dim: 4096,
  n_heads: 32,
  n_blocks: 32,
  hidden_dim: 11008,
  dtype: 'bf16',
};

const llama3Config: Omit<Llama3Config, 'model'> = {
  // 1B Llama3
  epochs: 1,
  batch_size: 1,
  vocab_size: 128_258,
  context_length: 256, // 131_072,
  stride: 256, // == context_length
  emb_dim: 2048,
  n_heads: 32,
  n_blocks: 16,
  hidden_dim: 8192,
  n_kv_groups: 8,
  rope_base: 500_000.0,
  rope_freq: {
    factor: 32.0,
    low_freq_factor: 1.0,
    high_freq_factor: 4.0,
    original_context_length: 256, // 8192
  },
  dtype: 'bf16',
};

const qwen3Config: Omit<Qwen3Config, 'model'> = {
  // 0.6B Qwen3
  epochs: 1,
  batch_size: 1,
  vocab_size: 151_669, // 151_936,
  context_length: 256, // 40_960
  stride: 256, // == context_length
  emb_dim: 1024,
  n_heads: 16,
  n_blocks: 28,
  hidden_dim: 3072,
  head_dim: 128,
  qk_norm: true,
  n_kv_groups: 8,
  rope_base: 1_000_000.0,
  dtype: 'bf16',
};

const smollm3Config: Omit<SmolLM3Config, 'model'> = {
  // 0.6B SmolLM3
  epochs: 1,
  batch_size: 1,
  vocab_size: 151_669, // 151_936
  context_length: 256, // 40_960
  stride: 256, // == context_length
  emb_dim: 1024,
  n_heads: 16,
  n_blocks: 28,
  hidden_dim: 3072,
  dtype: 'bf16',
};

export const modelConfigs: Record<ModelType, Omit<ModelConfig, 'model'>> = {
  'GPT-2': gpt2Config,
  Llama2: llama2Config,
  Llama3: llama3Config,
  Qwen3: qwen3Config,
  SmolLM3: smollm3Config,
};
