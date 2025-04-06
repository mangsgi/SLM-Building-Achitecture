import React from 'react';
import { EditField, ReadField, EditSelectField } from './Components';

export interface FieldConfig {
  type: 'text' | 'number' | 'select';
  label: string;
  name: string;
  value: string;
  placeholder?: string;
  options?: string[]; // for select
}

interface FieldRendererProps {
  fields: FieldConfig[];
  editMode: boolean;
  onChange: (name: string, value: string) => void;
}

const FieldRenderer: React.FC<FieldRendererProps> = ({
  fields,
  editMode,
  onChange,
}) => {
  return (
    <div>
      {fields.map((field) => {
        if (editMode) {
          if (field.type === 'select') {
            return (
              <EditSelectField
                key={field.name}
                label={field.label}
                id={field.name}
                name={field.name}
                value={field.value}
                onChange={(value) => onChange(field.name, value)}
                options={field.options || []}
              />
            );
          } else {
            return (
              <EditField
                key={field.name}
                label={field.label}
                id={field.name}
                name={field.name}
                value={field.value}
                placeholder={field.placeholder}
                onChange={(value) => onChange(field.name, value)}
              />
            );
          }
        } else {
          return (
            <ReadField
              key={field.name}
              label={field.label}
              value={field.value}
            />
          );
        }
      })}
    </div>
  );
};

export default FieldRenderer;
