-- function Header(el)
--   return {el, pandoc.Para{pandoc.Str('-----')}}
-- end

function Header(el)
  local text = pandoc.utils.stringify(el)
  return pandoc.Plain{
    pandoc.Str(text),
    pandoc.LineBreak(),
    pandoc.Str('-----')
  }
end
