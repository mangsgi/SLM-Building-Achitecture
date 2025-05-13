import React from 'react';

const CanvasHamburgerButton: React.FC = () => {
  return (
    <button className="w-[33.6px] h-[33.6px] flex flex-col items-center justify-center gap-[6px] bg-[rgb(0,0,0)] rounded-[10px] cursor-pointer border-0 group">
      <span
        className="w-1/2 h-[2px] bg-[rgb(229,229,229)] flex items-center justify-center relative rounded-[2px]
    before:content-[''] before:w-[2px] before:h-[2px] before:bg-[rgb(126,117,255)] before:absolute before:rounded-full before:border-2 before:border-white before:transition-all before:duration-300 before:shadow-[0_0_5px_white]
    before:-translate-x-[4px]
    group-hover:before:translate-x-[4px]"
      ></span>
      <span
        className="w-1/2 h-[2px] bg-[rgb(229,229,229)] flex items-center justify-center relative rounded-[2px]
    before:content-[''] before:w-[2px] before:h-[2px] before:bg-[rgb(126,117,255)] before:absolute before:rounded-full before:border-2 before:border-white before:transition-all before:duration-300 before:shadow-[0_0_5px_white]
    before:translate-x-[4px]
    group-hover:before:-translate-x-[4px]"
      ></span>
      <span
        className="w-1/2 h-[2px] bg-[rgb(229,229,229)] flex items-center justify-center relative rounded-[2px]
    before:content-[''] before:w-[2px] before:h-[2px] before:bg-[rgb(126,117,255)] before:absolute before:rounded-full before:border-2 before:border-white before:transition-all before:duration-300 before:shadow-[0_0_5px_white]
    before:-translate-x-[4px]
    group-hover:before:translate-x-[4px]"
      ></span>
    </button>
  );
};

export default CanvasHamburgerButton;
