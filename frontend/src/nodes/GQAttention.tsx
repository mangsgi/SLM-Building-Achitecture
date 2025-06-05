import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';

import { NodeTitle } from './components/Components';
import { GQAttentionData } from './components/NodeData';
import { LayerWrapper } from './components/LayerWrapper';
import NodeActionPanel from './components/ActionPanel';
import NodeInfoModal from './components/NodeInfoModal';
import { useCommonNodeActions } from './useCommonNodeActions';
import FieldRenderer, { FieldConfig } from './components/FieldRenderer';
import { nodeInfo, nodeFieldInfo } from './components/nodeInfo';

const getFields = (data: GQAttentionData): FieldConfig[] => [
  {
    type: 'number',
    label: 'Number of Heads:',
    name: 'numHeads',
    value: data.numHeads?.toString() || '',
    placeholder: 'Enter number of heads',
    info: nodeFieldInfo.gqAttention.numHeads,
  },
  {
    type: 'number',
    label: 'Context Length:',
    name: 'ctxLength',
    value: data.ctxLength?.toString() || '',
    placeholder: 'Enter context length',
    info: nodeFieldInfo.gqAttention.ctxLength,
  },
  {
    type: 'number',
    label: 'Dropout Rate:',
    name: 'dropoutRate',
    value: data.dropoutRate?.toString() || '',
    placeholder: 'Enter dropout rate',
    info: nodeFieldInfo.gqAttention.dropoutRate,
  },
  {
    type: 'select',
    label: 'QKV Bias:',
    name: 'qkvBias',
    value: data.qkvBias?.toString() || 'false',
    options: ['true', 'false'],
    info: nodeFieldInfo.gqAttention.qkvBias,
  },
];

interface GQAttentionLayerProps {
  id: string;
}

export const GQAttentionLayer: React.FC<GQAttentionLayerProps> = ({ id }) => {
  const { setNodes, getNode, setEdges } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [isInfoOpen, setIsInfoOpen] = useState<boolean>(false);
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

  const node = getNode(id);
  if (!node) return null;
  const currentData = node.data as GQAttentionData;

  const handleFieldChange = (field: keyof GQAttentionData, value: string) => {
    const stringFields: (keyof GQAttentionData)[] = ['label'];
    const newValue = stringFields.includes(field) ? value : Number(value);
    setNodes((nds) =>
      nds.map((nodeItem) => {
        if (nodeItem.id === id) {
          return {
            ...nodeItem,
            data: {
              ...nodeItem.data,
              [field]: newValue,
            },
          };
        }
        return nodeItem;
      }),
    );
  };

  const {
    handleDeleteClick,
    handleInfoClick,
    handleEditClick,
    handleSaveClick,
    handleNodeClick,
  } = useCommonNodeActions<GQAttentionData>({
    id,
    setNodes,
    setEditMode,
    setIsCollapsed,
    setEdges,
  });

  return (
    <LayerWrapper hideHandles={currentData.hideHandles}>
      <div className="relative group">
        <NodeTitle onClick={handleNodeClick}>{currentData.label}</NodeTitle>
        <NodeActionPanel
          editMode={editMode}
          onInfo={handleInfoClick}
          onEdit={handleEditClick}
          onSave={handleSaveClick}
          onDelete={handleDeleteClick}
        />
        {!isCollapsed && (
          <FieldRenderer
            fields={getFields(currentData)}
            editMode={editMode}
            onChange={(name: string, value: string) =>
              handleFieldChange(name as keyof GQAttentionData, value)
            }
            onInfoClick={(info) => {
              const event = new CustomEvent('fieldInfo', { detail: info });
              window.dispatchEvent(event);
            }}
          />
        )}
      </div>

      <NodeInfoModal isOpen={isInfoOpen} onClose={() => setIsInfoOpen(false)}>
        <h3 className="text-lg font-semibold mb-2">
          {nodeInfo.gqAttention.title}
        </h3>
        <p className="text-sm">{nodeInfo.gqAttention.description}</p>
      </NodeInfoModal>
    </LayerWrapper>
  );
};

export default GQAttentionLayer;
