-- Strip links
function Link(el)
    return el.content
end

-- Strip formatting
function Strong(el)
  return el.content
end

function Emph(el)
  return el.content
end

-- Strip images
function Image(el)
    return {}
end

-- Strip footnotes
function Note(el)
  return {}
end

-- Strip tables
function Table(el)
  return {}
end

-- Strip HTML
function RawInline(el)
  if el.format:match("html") then
    return {}
  end
end

function RawBlock(el)
  if el.format:match("html") then
    return {}
  end
end
