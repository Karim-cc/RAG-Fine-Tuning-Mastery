import React, { useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface CodeBlockProps {
  language: string;
  code: string;
  title?: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ language, code, title }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-4 rounded-lg overflow-hidden border border-slate-700 bg-slate-900 shadow-sm">
      <div className="flex justify-between items-center bg-slate-800 px-4 py-2 border-b border-slate-700">
        <span className="text-xs font-mono text-slate-300 uppercase">{language} {title && `| ${title}`}</span>
        <button 
          onClick={handleCopy}
          className="text-slate-400 hover:text-white transition-colors"
          title="Copy Code"
        >
          {copied ? <Check size={16} className="text-emerald-500" /> : <Copy size={16} />}
        </button>
      </div>
      <div className="p-4 overflow-x-auto">
        <pre className="font-mono text-sm text-slate-200">
          <code>{code}</code>
        </pre>
      </div>
    </div>
  );
};

export default CodeBlock;