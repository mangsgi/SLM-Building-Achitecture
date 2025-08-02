import React from 'react';
import { EditField, ReadField, EditSelectField } from './FieldComponents';

export interface FieldConfig {
  type: 'text' | 'number' | 'select';
  label: string;
  name: string;
  value: string;
  placeholder?: string;
  options?: string[]; // for select
  info?: {
    title: string;
    description: string;
  };
}

interface FieldRendererProps {
  fields: FieldConfig[];
  editMode: boolean;
  onChange: (name: string, value: string) => void;
  onInfoClick: (info: { title: string; description: string }) => void;
}

const FieldRenderer: React.FC<FieldRendererProps> = ({
  fields,
  editMode,
  onChange,
  onInfoClick,
}) => {
  return (
    <div>
      {fields.map((field) => {
        const fieldContent = editMode ? (
          field.type === 'select' ? (
            <EditSelectField
              key={field.name}
              label={field.label}
              id={field.name}
              name={field.name}
              value={field.value}
              onChange={(value) => onChange(field.name, value)}
              options={field.options || []}
              info={field.info}
              onInfoClick={onInfoClick}
            />
          ) : (
            <EditField
              key={field.name}
              label={field.label}
              id={field.name}
              name={field.name}
              value={field.value}
              placeholder={field.placeholder}
              onChange={(value) => onChange(field.name, value)}
              info={field.info}
              onInfoClick={onInfoClick}
            />
          )
        ) : (
          <ReadField
            key={field.name}
            label={field.label}
            value={field.value}
            info={field.info}
            onInfoClick={onInfoClick}
          />
        );

        return (
          <div key={field.name} className="relative">
            {fieldContent}
          </div>
        );
      })}
    </div>
  );
};

export default FieldRenderer;
