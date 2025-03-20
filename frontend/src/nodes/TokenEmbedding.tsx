import React, { useState } from 'react';
import { useReactFlow } from 'reactflow';
import { NodeData } from './NodeData';

export const TokenEmbeddingLayer: React.FC<{ data: NodeData }> = ({
  data: initialData,
}) => {
  const { setNodes } = useReactFlow();
  const [editMode, setEditMode] = useState<boolean>(false);
  const [vocabSizeStr, setInDimStr] = useState<string>(
    initialData.inDim.toString(),
  );
  const [embDimStr, setOutDimStr] = useState<string>(
    initialData.outDim.toString(),
  );

  // Edit 버튼 클릭
  const handleEditClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    console.log('Edit button clicked');
    setEditMode(true);
  };

  // Save 버튼 클릭
  const handleSaveClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    const newVocabSize =
      vocabSizeStr === '' ? initialData.inDim : Number(vocabSizeStr);
    const newEmbDim = embDimStr === '' ? initialData.outDim : Number(embDimStr);

    setEditMode(false);

    // 노드 데이터 업데이트
    if (initialData.id) {
      setNodes((nds) =>
        nds.map((node) => {
          if (node.id === initialData.id) {
            return {
              ...node,
              data: {
                ...node.data,
                inDim: newVocabSize,
                outDim: newEmbDim,
              },
            };
          }
          return node;
        }),
      );
    }
  };

  return (
    <div
      className="node border p-2 m-2 bg-white"
      style={{ pointerEvents: 'auto' }}
    >
      <h3 className="font-bold">{initialData.label}</h3>
      {editMode ? (
        <div>
          <div>
            <label htmlFor="vocabSizeInput" className="text-sm">
              Vocabulary Size:
            </label>
            <input
              id="vocabSizeInput"
              name="vocabSize"
              type="number"
              value={vocabSizeStr}
              placeholder="Enter Vocabulary Size"
              onChange={(e) => setInDimStr(e.target.value)}
              className="border rounded p-1 text-sm"
            />
          </div>
          <div>
            <label htmlFor="embDimSize" className="text-sm">
              Embedding Dimension Size:
            </label>
            <input
              id="embDimSize"
              name="embDim"
              type="number"
              value={embDimStr}
              placeholder="Enter embedding dimension"
              onChange={(e) => setOutDimStr(e.target.value)}
              className="border rounded p-1 text-sm"
            />
          </div>
          <button
            onClick={handleSaveClick}
            className="mt-2 px-2 py-1 bg-green-200 rounded text-sm"
          >
            Save
          </button>
        </div>
      ) : (
        <div>
          <p className="text-sm">
            Vocabulary Size: {vocabSizeStr || 'Not set'}
          </p>
          <p className="text-sm">
            Embedding Dimension Size: {embDimStr || 'Not set'}
          </p>
          <button
            onClick={handleEditClick}
            className="mt-2 px-2 py-1 bg-blue-200 rounded text-sm"
          >
            Edit
          </button>
        </div>
      )}
    </div>
  );
};

export default TokenEmbeddingLayer;
