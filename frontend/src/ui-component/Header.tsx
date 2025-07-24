import { useNavigate } from 'react-router-dom';

interface HeaderProps {
  children?: React.ReactNode;
}

function Header({ children }: HeaderProps) {
  const navigate = useNavigate();

  return (
    <header className="bg-white p-4 shadow flex justify-between items-center">
      <h1
        className="text-2xl font-semibold text-left cursor-pointer"
        onClick={() => navigate('/canvas')}
      >
        Building Your Own SLM
      </h1>
      <div>{children}</div>
    </header>
  );
}

export default Header;
