import globals from 'globals';
import pluginJs from '@eslint/js';
import tsEslint from 'typescript-eslint';
import tsParser from '@typescript-eslint/parser'; //languageOptions의 parser에서 오류 발생, parser 객체 import
import pluginReact from 'eslint-plugin-react';
import prettierPlugin from 'eslint-config-prettier';

/** @type {import('eslint').Linter.FlatConfig[]} */
export default [
  // 기본 ESLint 추천 규칙
  pluginJs.configs.recommended,
  // TypeScript 추천 규칙, 반드시 react/react-in-jsx-scope 설정보다 앞에 와야 rule이 적용됨.
  ...tsEslint.configs.recommended,
  // React Flat Config 추천 규칙
  pluginReact.configs.flat.recommended,
  // 공통 설정: 파일 범위, 파서 옵션, 전역 변수 등
  {
    files: ['**/*.{js,ts,jsx,tsx}'],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: 2021,
        sourceType: 'module',
        ecmaFeatures: {
          jsx: true,
        },
      },
      globals: { ...globals.browser, ...globals.node },
    },
    plugins: {
      react: pluginReact,
      prettier: prettierPlugin,
    },
    rules: {
      // React 17+에서는 JSX 사용 시 React 임포트 불필요
      'react/react-in-jsx-scope': 'off',
      'react/jsx-uses-react': 'off',
      '@typescript-eslint/no-explicit-any': 'off', // <- any 사용 허용
      'react/prop-types': 'off', // <- prop-types 사용 허용
      'no-unexpected-multiline': 'off', // <- 객체와 대괄호 ([) 사이에 줄바꿈 허용
    },
    settings: {
      react: {
        version: 'detect',
      },
    },
  },
];
